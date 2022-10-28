'''
Scrapes all images from the Generated Website
'''
from bs4 import BeautifulSoup
import os
from requests import get

'''
Helper function to create folders for each key word
'''
def create_folders(img_path, key_words):
    for word in key_words:
        try:
            os.mkdir(img_path + '/' + word)
        except OSError as error:
            continue

'''
Generate all paths for a specific prompt
'''
def find_all_prompts(needed_divs):
    prompts = []
    after_artists = False
    for div in needed_divs:
        mod_string = div.string.replace(",", "")
        str_list = mod_string.split(" ")
        new_string = ""
        for i in range(len(str_list)):
            if (i == 0):
                new_string += (str_list[i].lower())
            else:
                new_string += (str_list[i][0].upper() + str_list[i][1:])

        if (new_string == "impressionism"):
            after_artists = True
        if (after_artists):
            if ("painting" in new_string):
                if (new_string == "airbrush(stencilArt)Painting"):
                    prompts.append("airbrushPaintingStencilArt")
                else:
                    prompts.append(new_string)
            else:
                prompts.append(new_string + "Painting")
    return prompts

'''
Find all possible path names we can take with 'prompts'
'''
def scrape_images():
    url = 'https://generrated.com/'
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    needed_divs = soup.find_all("div", class_="c-iTLBvH")
    return needed_divs

'''
Save all images in mass
'''
def save_all_images(prompts):
    create_folders('artist-images', prompts)
    for prompt in prompts:
        prompt_url = 'https://generrated.com/prompts/' + str(prompt)
        response = get(prompt_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        imgs = soup.find_all("img")
        counter = 0
        for img in imgs:
            img_src = (img.get('src'))
            print(img_src)
            if "/generated_images" in img_src:
                img_data = get("https://generrated.com/" + img_src).content
                with open('artist-images/' + prompt + '/' + prompt + '-' + str(counter) + '.jpg', 'wb') as handler:
                    handler.write(img_data)
                counter += 1

'''
Runs all of the functions to generate images
'''
def main():
    needed_divs = scrape_images()
    prompts = find_all_prompts(needed_divs)
    save_all_images(prompts)

if __name__ == '__main__':
    main()