import timeago
from datetime import datetime
import pytz
from urllib.parse import unquote, urlparse

from pyramid.httpexceptions import HTTPServiceUnavailable, HTTPFound
from pyramid.view import view_config, exception_view_config
from requests import RequestException, HTTPError

from ..utils.listparser import render_thread, MailboxNotFound

ERROR_QUEUE = "ERRORS"


@view_config(route_name="home", request_method="GET",
             renderer='pml:templates/home.jinja2')
def home_page(request):
    errors = request.session.pop_flash(queue=ERROR_QUEUE)
    return {"errors": errors}


@view_config(route_name="view-thread", request_method="GET",
             renderer='pml:templates/thread.jinja2')
def view_thread(request):
    src = unquote(request.GET.get("src"))
    try:
        parsed = urlparse(src)
        assert parsed.netloc == "lore.kernel.org"
    except (AttributeError, AssertionError):
        request.session.flash("Not a valid mailing list URL", queue=ERROR_QUEUE)
        return HTTPFound(location=request.route_url('home'))

    thread = render_thread(src)
    return {
        "title": thread.subject,
        "contents": thread.content,
        "original": src,
        "cached": thread.cached,
        "pull_time": thread.get_pull_time().isoformat(),
        "pull_time_str": timeago.format(thread.get_pull_time(),
                                        datetime.utcnow().replace(tzinfo=pytz.UTC))
    }


@exception_view_config(RequestException, route_name='view-thread')
@exception_view_config(MailboxNotFound, route_name='view-thread')
def failed_upstream(exc, request):
    url = request.route_url('home')
    if isinstance(exc, MailboxNotFound):
        msg = "Did not find a valid mailing list thread at the URL"
    elif isinstance(exc, HTTPError):
        msg = "Error in fetching the resource, make sure you're passing a valid URL"
    else:
        msg = "Error connecting to the resource, please try again later."
    return HTTPServiceUnavailable(location=url, detail=msg)
