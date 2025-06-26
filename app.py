import tkinter as tk
import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImage
from authtoken import auth_token

import torch    # using PyTorch
from diffusers import StableDiffusionPipeline   # diffuser library
import threading
import os
import time

# Create the app window
app = tk.Tk()
app.geometry("532x680")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Prompt input
prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Progress bar
progress = ctk.CTkProgressBar(master=app, width=512)
progress.place(x=10, y=60)
progress.set(0)
progress.stop()
progress.place_forget()  # Hidden initially

# Image display label
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Load the model on CPU
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=auth_token
)
pipe.to("cpu")

# Generation logic in background thread
def generate():
    def run_generation():
        prompt_text = prompt.get().strip()
        if len(prompt_text) > 300:
            prompt_text = prompt_text[:300]

        progress.place(x=10, y=60)
        progress.start()

        try:
            with torch.autocast("cpu"):
                result = pipe(prompt_text, num_inference_steps=15, guidance_scale=7.5)
                image = result["images"][0]

                # Save to desktop
                desktop = os.path.join(os.path.join(os.environ["USERPROFILE"]), "Desktop")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"generated_{timestamp}.png"
                image_path = os.path.join(r"C:\Users\User\Downloads\Stable Diffusion", filename)
                image.save(image_path)

                # Preview in app
                img = CTkImage(light_image=image, size=(512, 512))
                lmain.configure(image=img)
                lmain.image = img
        finally:
            progress.stop()
            progress.place_forget()

    threading.Thread(target=run_generation).start()

# Generate button
trigger = ctk.CTkButton(master=app, height=40, width=120, text="Generate", text_color="white", fg_color="blue", command=generate)
trigger.place(x=206, y=600)

# Run the app
app.mainloop()