import json
import codecs

file_path = "data/sitemap.json"

with open(file_path, "r") as f:
    lines = f.readlines()
    raw_data = [json.loads(line) for line in lines]
    
ingredients_file_path = "data/ingredients"
recipes_file_path = "data/recipes"

ingredients_file = codecs.open(ingredients_file_path, "w", "utf-8-sig")
recipes_file = codecs.open(recipes_file_path, "w", "utf-8-sig")

for recipe in raw_data:
    recipe_name = recipe["name"] + "\n"
    ingredient_names = " ".join([i[0] for i in recipe["ingredients"]]) + "\n"
    
    ingredients_file.write(ingredient_names)
    recipes_file.write(recipe_name)
    
ingredients_file.flush()
ingredients_file.close()

recipes_file.flush()
recipes_file.close()