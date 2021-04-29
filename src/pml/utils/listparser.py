# this file should ideally not have "pyramid" related imports
import contextlib
import email
import gzip
import html
import itertools
import json
import mailbox
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from email.header import decode_header
from email.utils import parseaddr
from typing import Dict, List, Optional, Set, Generator, Tuple, Union
from urllib.parse import urljoin, urlparse

import pytz
import redis
import requests
import timeago
from bs4 import BeautifulSoup
from dataclasses import dataclass
from pygments import highlight, lexers
from pygments.formatters import HtmlFormatter
from pyramid.renderers import render


class MailboxNotFound(Exception):
    pass


@dataclass
class MultiLevelMessage:
    message_id: str
    message: Optional[mailbox.mboxMessage]
    timestamp: Optional[datetime]
    children: List
    level: int
    num_children: int


EXTRA_TZ_RE = re.compile(r"\([A-Z]+\)")


def clean_dt(dt):
    # some dates end with (AEST), (CEST) etc.
    return EXTRA_TZ_RE.sub("", dt).strip()


def parse_date_field(dt):
    try:
        return datetime.strptime(
            clean_dt(dt).split(",")[-1].strip(), "%d %b %Y %H:%M:%S %z"
        ).astimezone(pytz.UTC)
    except ValueError:
        return datetime.utcnow().astimezone(pytz.UTC)


def parse_timestamp(item):
    return item.timestamp or parse_date_field("")


def iter_thread(
    message_id: str,
    thread: Dict[str, Set[str]],
    message_map: Dict[str, mailbox.mboxMessage],
    level: int = 0,
    update_child: callable = None,
):
    replies = thread.get(message_id)  # list of message-ids
    if not replies:
        children = []
    else:
        children = [
            iter_thread(r, thread, message_map,
                        level=level + 1,
                        update_child=update_child)
            for r in replies
        ]

    num_children = sum(_.num_children for _ in children) if children else 0

    if message_id and update_child is not None:
        update_child(message_id)

    msg = message_map.get(message_id)
    if not msg:
        date = min(_.timestamp for _ in children) if children else None
    else:
        date = parse_date_field(msg.get('date'))

    return MultiLevelMessage(
        message_id,
        msg,
        date,
        list(sorted(children, key=parse_timestamp)),
        level,
        num_children + len(children),
    )


def tree_from_references(messages):
    """
    Generate mapping of <message-id>:[<immediate-responses>]
    using the 'References' field in each message.
    Accounts for all messages, even [not found] ones
    """
    tree = defaultdict(set)
    for msg in messages:
        if not msg.get('references'):
            tree[msg['message-id']] |= set()
            continue

        msg_refs = msg['references'].split() + [msg['message-id']]
        for parent, child in zip(msg_refs[:-1], msg_refs[1:]):
            tree[parent].add(child)
    return tree


def get_top_level_sort_fn(message_map):
    def fn(message_id):
        try:
            return parse_date_field(message_map[message_id]['date'])
        except KeyError:
            return parse_date_field("")
    return fn


def generate_thread(messages: List[mailbox.mboxMessage],
                    update_child: callable = None):
    # this and iter_thread can probably benefit
    # from being implemented as a class
    tree = tree_from_references(messages)
    message_map = {_.get('message-id'): _ for _ in messages}

    # get parent-level message-ids that do not exist as children
    top_level = set(tree.keys()) - set(
        itertools.chain.from_iterable(tree.values())
    )

    sort_fn = get_top_level_sort_fn(message_map)
    for message_id in sorted(top_level, key=sort_fn):
        yield iter_thread(message_id, tree, message_map,
                          update_child=update_child)


class BlockType:
    Text = "text"
    Quote = "quote"
    Diff = "diff"


def parse_payload(payload: str) -> Generator[Tuple[str, str], None, None]:
    text = payload.strip()
    current_type = None
    lines = []
    for line in text.split("\n"):
        # warning: hack! Couldn't figure out a way to reliable determine end of diff
        # Most messages I saw didn't have text beyond the diff, except a signature (--)
        # I think there is enough information to determine this within the diff
        # itself (.e.g "@@ -123,3 +123,4") but it will require some advanced parsing
        if current_type == BlockType.Diff and (line.strip() != "--"
                                               and not line.startswith(">")):
            line_type = BlockType.Diff
        else:
            if line.startswith(">"):
                line_type, line = BlockType.Quote, line[1:]
            elif line.startswith("diff "):
                line_type = BlockType.Diff
            else:
                line_type = BlockType.Text

        if current_type is None or line_type == current_type:
            lines.append(line)
            current_type = line_type
        else:  # elif  line_type != current_type:
            yield current_type, render_block(lines, current_type)
            current_type, lines = line_type, [line]

    yield current_type, render_block(lines, current_type)


html_formatter = HtmlFormatter()
diff_lexer = lexers.get_lexer_by_name("diff")


def render_block(lines, block_type):
    if block_type in (BlockType.Text, BlockType.Quote):
        return "<br>".join([html.escape(_) for _ in lines])
    elif block_type == BlockType.Diff:
        return highlight(
            "\n".join(lines),
            diff_lexer,
            html_formatter,
        )


SPECIALS = ["kernel.org", "linuxfoundation.org"]


def decode_payload(payload, charsets=('utf-8', 'latin-1')):
    for c in charsets:
        try:
            return payload.decode(c)
        except UnicodeDecodeError as e:
            continue
    return payload.decode(errors="replace")


def get_message_body(
        message: Union[mailbox.mboxMessage, email.message.Message],
):
    if message.is_multipart():
        payload = message.get_payload()
        return "".join(get_message_body(_) for _ in payload) if payload else ""

    if message.get_content_type() == "text/plain":
        return decode_payload(message.get_payload(decode=True))
    else:
        return f'\n[View original post for attachment ({message.get_filename()})]'


def parse_header(header_field) -> str:
    if not header_field:
        return ""

    return "".join([
        content.decode() if isinstance(content, bytes) else content
        for content, encoding in decode_header(header_field)
    ])


def generate_inner_html(m: MultiLevelMessage):
    if m.children:
        children = "".join(generate_inner_html(_) for _ in m.children)
    else:
        children = ""

    if m.message:
        name, addr = parseaddr(parse_header(m.message.get('from')))
        _, domain = addr.split("@")
        payload = get_message_body(m.message)
        raw_timestamp = m.timestamp
    else:
        name, addr, domain, raw_timestamp, payload = (
            "[deleted]", "", "", m.timestamp, "[removed]"
        )

    formatted_ts = (
        timeago.format(raw_timestamp, datetime.utcnow().astimezone(pytz.UTC))
        if raw_timestamp else None
    )
    context = {
        "id": m.message_id,
        "from": addr,
        "name": name,
        "timestamp": formatted_ts,
        "raw_timestamp": raw_timestamp.isoformat() if raw_timestamp else None,
        "children": children,
        "count": m.num_children,
        "payload": list(parse_payload(payload)) if payload else [],
        "special": domain if domain.lower() in SPECIALS else None,
        "edu": domain if domain.endswith(".edu") else None,
        "missing": m.message is None
    }
    return render('pml:templates/post.jinja2', context)


def get_mbox_url(thread_url):
    # warning: only supports lore.kernel.org lists
    page = requests.get(thread_url)
    page.raise_for_status()
    parsed = BeautifulSoup(page.content, "html.parser")
    mbox_tag = parsed.find("a", string="mbox.gz")
    mbox_url = mbox_tag.get("href") if mbox_tag else None
    if not mbox_url:
        raise MailboxNotFound()
    return urljoin(thread_url, mbox_url)


@contextlib.contextmanager
def mbox_from_url(url):
    temp_path = os.path.join("/tmp", f"{str(uuid.uuid4())}.mbox")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with open(temp_path, "wb") as f:
            f.write(gzip.decompress(resp.content))
        yield mailbox.mbox(temp_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(temp_path)


@dataclass
class Thread:
    subject: str
    to: str
    cc: str
    content: str
    cached: bool = False
    pull_time: float = None  # seconds since epoch

    def get_pull_time(self):
        return datetime.fromtimestamp(self.pull_time, tz=pytz.UTC)


CACHE_EXPIRY = 3600  # in seconds


def get_update_child_callable(redis_client: redis.Redis, root_message_id: str):
    def cb(message_id):
        redis_client.set(message_id, root_message_id, ex=CACHE_EXPIRY)
    return cb


def get_message_id_from_url(thread_url) -> Optional[str]:
    try:
        parsed = urlparse(thread_url)
        return f"<{parsed.path.strip('/').split('/')[1]}>"
    except IndexError:
        return


def format_root_id(message_id):
    return f"content::{message_id}"


def get_cached_content(redis_client: redis.Redis, message_id: str = None):
    if not message_id:
        return

    root_message_id = redis_client.get(message_id)
    if not root_message_id:
        return

    return redis_client.get(format_root_id(root_message_id.decode()))


def render_thread(thread_url, use_cached=True) -> Thread:
    redis_client = redis.Redis()
    message_id = get_message_id_from_url(thread_url)
    cached = get_cached_content(redis_client, message_id)
    if cached and use_cached:
        thread = Thread(**json.loads(cached))
        thread.cached = True
        return thread

    mbox_url = get_mbox_url(thread_url)
    with mbox_from_url(mbox_url) as mbox:
        first = mbox[0]
        root_message_id = first.get('message-id')
        cb = get_update_child_callable(redis_client, root_message_id)
        items = generate_thread(mbox.values(), cb)
        content = "".join(generate_inner_html(_) for _ in items)

    thread = Thread(
        first.get('subject'),
        parse_header(first.get('to')),
        parse_header(first.get('cc')),
        content,
        pull_time=time.time(),
    )
    redis_client.set(format_root_id(root_message_id), json.dumps(thread.__dict__),
                     ex=CACHE_EXPIRY)
    return thread
