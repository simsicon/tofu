# -*- coding: utf-8 -*-
import scrapy
from scrapy.loader import ItemLoader
from tofu.items import RecipeItem

class SitemapSpider(scrapy.spiders.SitemapSpider):
    name = "sitemap"
    sitemap_urls = ['http://www.xiachufang.com/sitemap.xml']
    sitemap_rules = [
        ('/recipe/', 'parse_recipe')
    ]

    def parse_recipe(self, response):
        name = response.css(".page-title::text").extract()[0].strip()
        url = response.url
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

        recipe = RecipeItem(name=name, url=url)
        recipe['image_urls'] = image_urls
        recipe['description'] = description
        recipe['ingredients'] = ingredients
        recipe['steps'] = steps
        recipe['tips'] = tips

        yield recipe
