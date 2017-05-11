from random import choice
from scrapy import signals
from scrapy.exceptions import NotConfigured
from urllib import urlopen
import pdb
import time
import global_variables
import logging


class RotateUserAgentMiddleware(object):
    """Rotate user-agent for each request."""
    def __init__(self, user_agents):
        self.user_agents = user_agents

    @classmethod
    def from_crawler(cls, crawler):
        user_agents = crawler.settings.get('USER_AGENT_CHOICES', [])

        if not user_agents:
            raise NotConfigured("USER_AGENT_CHOICES not set or empty")

        o = cls(user_agents)
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)

        return o

    def spider_opened(self, spider):
        return

    def process_request(self, request, spider):
        if not self.user_agents:
            return

        request.headers['user-agent'] = choice(self.user_agents)

class RandomProxyMiddleware(object):
    def process_request(self, request, spider):
        proxy_fetched_at = global_variables.PROXY_FETCHED_AT
        past_timespan = int(time.time()) - proxy_fetched_at
        if len(global_variables.PROXY_URL) == 0 or past_timespan > 5:
            proxy_fetch_url = "PROXY_LIST_URL"
            global_variables.PROXY_URL = urlopen(proxy_fetch_url).read().strip()
            global_variables.PROXY_FETCHED_AT = int(time.time())
            logging.log(logging.INFO, "Proxy IP:{}".format(global_variables.PROXY_URL))

        request.meta['proxy'] = 'http://{}'.format(global_variables.PROXY_URL)
