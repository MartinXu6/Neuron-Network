from bs4 import BeautifulSoup
import httpx

image_links = []
url = "https://www.freepik.com/photos/dog"
response = httpx.get(url)
print(response)
soup = BeautifulSoup(response.text, "html.parser")
for image_box in soup.select("div.showcase__thumbnail"):
    result = {
        "link": image_box.select_one("img").attrs["src"],
    }
    # Append each image and title to the result array
    image_links.append(result)
print(image_links)
for image_object in range(len(image_links)):
    # Create a new .png image file
    with open(f"./images/dog{image_object}.png", "wb") as file:
        image = httpx.get(image_links[image_object]["link"])
        # Save the image binary data into the file
        file.write(image.content)
        print(f"Image dog{image_object}.png has been scraped")
