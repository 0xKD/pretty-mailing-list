import os

from pyramid.config import Configurator
from pyramid.session import SignedCookieSessionFactory


def main(global_config, **settings):
    session_factory = SignedCookieSessionFactory(
        os.path.expandvars(settings.get('cookies.secret'))
    )
    config = Configurator(settings=settings,
                          session_factory=session_factory)
    config.include('pyramid_jinja2')
    config.include('.routes')
    config.scan()
    return config.make_wsgi_app()
