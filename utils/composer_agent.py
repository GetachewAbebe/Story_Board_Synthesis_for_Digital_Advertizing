from PIL import Image

class AutoGenImageComposer:
    def __init__(self, assets):
        self.assets = assets
        self.positions = [(0, 0)] * len(assets)  # Initialize positions

    def compose(self):
        composed_image = Image.new('RGB', (800, 600), (255, 255, 255))

        # Logo
        logo_img = Image.open(self.assets[0]).resize((100, 100)).convert('RGBA')
        composed_image.paste(logo_img, (50, 50), mask=logo_img)

        # Background
        background_img = Image.open(self.assets[1]).resize((800, 600)).convert('RGBA')
        composed_image.paste(background_img, (0, 0), mask=background_img)

        # CTA
        cta_img = Image.open(self.assets[2]).resize((200, 50)).convert('RGBA')
        composed_image.paste(cta_img, (300, 500), mask=cta_img)

        # Other components
        for i, asset in enumerate(self.assets[3:]):
            img = Image.open(asset).resize((100, 100)).convert('RGBA')
            x = 50 + (i % 3) * 150
            y = 200 + (i // 3) * 150
            composed_image.paste(img, (x, y), mask=img)

        return composed_image
    def adjust_composition(self, feedback):
        if "alignment" in feedback:
            self.positions = [(i * 110, 0) for i in range(len(self.assets))]
        elif "spacing" in feedback:
            self.positions = [(i * 120, 0) for i in range(len(self.assets))]
        elif "center" in feedback:
            self.positions = [(350 - (len(self.assets) * 100) // 2 + i * 100, 250) for i in range(len(self.assets))]
        elif "right" in feedback:
            self.positions = [(700 - i * 100, 250) for i in range(len(self.assets))]
        elif "random" in feedback:
            import random
            self.positions = [(random.randint(0, 700), random.randint(0, 500)) for _ in range(len(self.assets))]

# Define your image assets
assets = ['image1.jpg', 'image2.jpg', 'image3.jpg','image4.jpg', 'image5.jpg', 'image6.jpg']  # Replace with actual image file paths

# Create an instance of the composer
composer = AutoGenImageComposer(assets)

# Adjust the composition based on feedback
composer.adjust_composition("alignment")

# Compose the image
composed_image = composer.compose()

# Display the composed image
composed_image.show()

# Optionally, save the composed image to a file
composed_image.save('composed_image_alignment.png')