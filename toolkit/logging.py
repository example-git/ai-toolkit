from typing import List, OrderedDict, Optional
from PIL import Image, ImageDraw, ImageFont
import wandb

from toolkit.config_modules import LoggingConfig

# Base logger class
class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def commit(self, step: Optional[int] = None):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def log_all_images(self, *args, **kwargs):
        pass

    def finish(self):
        pass

# Wandb logger class
class WandbLogger(EmptyLogger):
    def __init__(self, project: str, run_name: str | None, config: OrderedDict) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config
        self.run = None

        self.logged_images: dict = {}  # Internal dict to store images mapped by id

    def start(self):
        try:
            # Initialize wandb run
            self.run = wandb.init(project=self.project, name=self.run_name, config=self.config)
            self._log = wandb.log  # log function
            self._image = wandb.Image  # image object
        except ImportError:
            raise ImportError("Failed to import wandb. Please install wandb by running `pip install wandb`")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize wandb: {e}")

    def log(self, *args, **kwargs):
        try:
            # Log data without incrementing the step
            self._log(*args, **kwargs, commit=False)
        except Exception as e:
            print(f"Logging Error: {e}")

    def commit(self, step: Optional[int] = None):
        try:
            # Commit the log by logging an empty dictionary with commit=False
            self._log({}, step=step, commit=True)
        except Exception as e:
            print(f"Commit Error: {e}")

    def log_image(
        self,
        image: Image,
        id: int,  # sample index
        caption: str | None = None,  # positive prompt
        **kwargs,
    ):
        """
        Logs an image to WandB with a given sample ID and optional caption.

        Args:
            image (Image): The PIL Image to log.
            id (int): The sample index.
            caption (str | None): An optional caption for the image.
            **kwargs: Additional keyword arguments for wandb.Image and wandb.log.
        """
        try:
            # Ensure the image is a PIL Image
            if not isinstance(image, Image.Image):
                raise TypeError("The 'image' parameter must be a PIL.Image.Image instance.")

            # Create a wandb Image object
            wandb_image = self._image(image, caption=caption, **kwargs)

            # Prepare the log entry
            log_entry = {
                f"sample_{id}": wandb_image,
                "caption": caption
            }

            wand_log = {
                f"sample_{id}": wandb_image,
            }

            # Combine log_entry with additional kwargs if any
            if kwargs:
                combined_log = wandb_log
            else:
                combined_log = wandb_log

            # Log the combined entry to wandb
            self._log(combined_log)

            # Store the wandb image mapped by its id
            self.logged_images[id] = wandb_image

        except Exception as e:
            print(f"Log Image Error: {e}")

    def add_caption_to_image(self, pil_image: Image.Image, caption: str) -> Image.Image:
        """
        Adds a caption to a PIL image.

        Args:
            pil_image (Image.Image): The image to add a caption to.
            caption (str): The caption text.

        Returns:
            Image.Image: The image with the caption added.
        """
        try:
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.load_default()
            text_width, text_height = draw.textsize(caption, font=font)
            width, height = pil_image.size
            # Define text position (bottom center)
            position = ((width - text_width) / 2, height - text_height - 10)
            # Add a semi-transparent rectangle behind the text for readability
            rectangle_height = text_height + 10
            rectangle = Image.new('RGBA', (text_width + 10, rectangle_height), (0, 0, 0, 150))
            pil_image.paste(rectangle, (int((width - text_width) / 2) - 5, int(height - rectangle_height)), rectangle)

            # Add text to image
            draw.text(position, caption, (255, 255, 255), font=font)
            return pil_image
        except Exception as e:
            print(f"Add Caption Error: {e}")
            return pil_image  # Return original image if captioning fails

    def aggregate_images_with_captions(self, grid_size: Optional[tuple] = (1, 6)) -> wandb.Image:
        """
        Aggregate all logged images into a single WandB Image object with captions embedded.

        Args:
            grid_size (Optional[tuple]): Tuple specifying the grid size (rows, cols).
                                         If None, images will be logged as a list.

        Returns:
            wandb.Image: The aggregated image object.
        """
        try:
            if not self.logged_images:
                raise ValueError("No images have been logged to aggregate.")

            # Embed captions into images
            pil_images_with_captions = [
                self.add_caption_to_image(wi.image.copy(), wi.caption or "")
                for wi in self.logged_images.values()
            ]

            if grid_size:
                rows, cols = grid_size
                if rows * cols < len(pil_images_with_captions):
                    raise ValueError("Grid size is too small for the number of images.")

                thumbnail_size = pil_images_with_captions[0].size
                grid_width = cols * thumbnail_size[0]
                grid_height = rows * thumbnail_size[1]

                grid_image = Image.new('RGB', (grid_width, grid_height))

                for idx, img in enumerate(pil_images_with_captions):
                    row = idx // cols
                    col = idx % cols
                    if row >= rows:
                        break
                    grid_image.paste(img, (col * thumbnail_size[0], row * thumbnail_size[1]))

                aggregated_image = self._image(grid_image, caption="Aggregated Image Grid with Captions")

            else:
                aggregated_image = self._image(
                    pil_images_with_captions,
                    caption="Aggregated Image List with Captions"
                )

            return aggregated_image
        except Exception as e:
            print(f"Aggregate Images with Captions Error: {e}")
            raise

    def log_all_images(
        self,
        img_list: list[tuple],
        **kwargs,
    ):
        """
        Logs an image without aggregation.

        Args:
            image (Image): The PIL Image to log.
            id (int): The sample index.
            caption (str | None): An optional caption for the image.
            **kwargs: Additional keyword arguments for wandb.log.
        """
        try:
            # Ensure the image is a PIL Image

            # Create a wandb Image object
            wandb_list = {}
            for index in range(0, len(img_list)):
                image, id, caption = img_list[index]
                # print(f"{image} - {id} - {caption}")
                if not isinstance(image, Image.Image):
                    print("The 'image' parameter must be a PIL.Image.Image instance.")
                    pass
                else:
                    wandb_image = self._image(image, caption=caption, **kwargs)
                    wandb_list[f"sample_{id}"] = wandb_image
                    #self.log_image(image, id, caption)
            

            # Log the entry to wandb  committing
            print('All Logs sent to WanDB')
            self._log(wandb_list, commit=False)

            # Optionally, store the wandb image if desired
            # Uncomment the line below if you wish to keep track of all images logged via log_all_image
            #self.logged_images[id] = wandb_image

        except Exception as e:
            print(f"Log All Image Error: {e}")

    def aggregate_images(self, grid_size: Optional[tuple] = (1, 6)) -> wandb.Image:
        """
        Aggregate all logged images into a single WandB Image object.
        Optionally, arrange them in a grid.

        Args:
            grid_size (Optional[tuple]): Tuple specifying the grid size (rows, cols).
                                         If None, images will be logged as a list.

        Returns:
            wandb.Image: The aggregated image object.
        """
        # Use the updated method with captions
        return self.aggregate_images_with_captions(grid_size=grid_size)

    def log_aggregated_images(self, grid_size: Optional[tuple] = (1, 6), caption: str | None = None):
        """
        Log all aggregated images as a single object to WandB.

        Args:
            grid_size (Optional[tuple]): Tuple specifying the grid size (rows, cols).
                                         If None, images will be logged as a list.
            caption (str | None): Optional caption for the aggregated image.
        """
        try:
            aggregated_image = self.aggregate_images_with_captions(grid_size=grid_size)
            # Update caption if provided
            if caption:
                aggregated_image = self._image(aggregated_image.image, caption=caption)
            # Log the aggregated image
            self._log({"aggregated_images": aggregated_image}, commit=False)
        except ValueError as e:
            print(f"Aggregation Error: {e}")
        except Exception as e:
            print(f"Log Aggregated Images Error: {e}")

    def finish(self):
        try:
            if self.run is not None:
                self.run.finish()
            else:
                print("WandB run was not initialized.")
        except Exception as e:
            print(f"Finish Run Error: {e}")

# create logger based on the logging config
def create_logger(logging_config: LoggingConfig, all_config: OrderedDict):
    if logging_config.use_wandb:
        project_name = logging_config.project_name
        run_name = logging_config.run_name
        return WandbLogger(project=project_name, run_name=run_name, config=all_config)
    else:
        return EmptyLogger()
