# -*- coding: utf-8 -*-
import scrapy
import urlparse
from scrapy.loader import ItemLoader
from tofu.items import RecipeItem

class RecipeSpider(scrapy.Spider):
    name = "recipe"
    allowed_domains = ["xiachufang.com"]
    start_urls = (
        'http://www.xiachufang.com/category/',
    )
    base_url = 'http://www.xiachufang.com'

    def parse(self, response):
        links = response.css("li a")
        category_links = self.select_category_links(links)

        for category_link in category_links:
            name = category_link.css("::text").extract_first()
            url = category_link.css("a::attr(href)").extract_first()
            yield scrapy.Request(urlparse.urljoin(self.base_url, url),
                                 callback=self.parse_recipes)

    def parse_recipes(self, response):
        links = response.css(".normal-recipe-list p.name a")
        recipe_links = [link for link in links if link.css("a::attr(href)").re(r'^/recipe/\d+')]

        for recipe_link in recipe_links:
            name = recipe_link.css("::text").extract_first()
            url = recipe_link.css("a::attr(href)").extract_first()
            recipe = RecipeItem(name=name, url=url)
            request = scrapy.Request(urlparse.urljoin(self.base_url, url),
                                 callback=self.parse_recipe)
            request.meta["recipe"] = recipe
            yield request

        if len(recipe_links) > 0 and response.css(".pager a.next"):
            next_page_url = response.css('.pager a.next::attr(href)').extract_first()
            yield scrapy.Request(urlparse.urljoin(self.base_url, next_page_url),
                                 callback=self.parse_recipes)


    def parse_recipe(self, response):
        recipe = response.meta["recipe"]
        image_urls = response.css(".cover.image img::attr(src)").extract()
        description = "\n".join(response.css(".desc::text").extract()).strip()

        ingredient_rows = response.css("tr[itemprop=recipeIngredient]")
        ingredients = []
        for row in ingredient_rows:
            if row.css("td.name a"):
                _name = row.css("td.name a::text").extract_first().strip()
            else:
                _name = row.css("td.name::text").extract_first().strip()

            _unit = row.css("td.unit::text").extract_first().strip()
            ingredients.append((_name, _unit))

        step_rows = response.css(".steps li[itemprop=recipeInstructions]")
        steps = []
        for row in step_rows:
            _text = row.css("p.text::text").extract_first().strip()
            _imageurl = row.css("img::attr(src)").extract_first()
            steps.append((_text, _imageurl))

        tips = "\n".join(response.css(".tip-container .tip::text").extract()).strip()

        recipe['image_urls'] = image_urls
        recipe['description'] = description
        recipe['ingredients'] = ingredients
        recipe['steps'] = steps
        recipe['tips'] = tips

        yield recipe

        explore_categories = response.css(".right-panel li a")
        if explore_categories:
            category_links = self.select_category_links(explore_categories)
            for category_link in category_links:
                name = category_link.css("::text").extract_first()
                url = category_link.css("a::attr(href)").extract_first()
                yield scrapy.Request(urlparse.urljoin(self.base_url, url),
                               callback=self.parse_recipes)

    def select_category_links(self, links):
        return [link for link in links if link.css("a::attr(href)").re(r'^/category/\d+')]
