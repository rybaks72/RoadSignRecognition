import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Message
from PIL import Image, ImageTk

from predict import annotate_image, predict_many


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Sign Recognition")
        self.file_path = None
        self.photo = None

        self.title_label = Label(root, text="Road Sign Recognition Application", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        self.authors_label = Label(root, text="Authors: Sylwia Rybak, Wojciech Sendek, Stanisław Zieliński, Jakub Szostak")
        self.authors_label.pack()

        self.controls_frame = Frame(root)
        self.controls_frame.pack(pady=10)

        self.upload_button = Button(self.controls_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=5)

        self.classify_button = Button(self.controls_frame, text="Classify Signs", command=self.classify_image, state=tk.DISABLED)
        self.classify_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = Button(self.controls_frame, text="Clear", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        self.result_label = Message(root, text="", font=("Helvetica", 12), width=650)
        self.result_label.pack(pady=10)

    def _show_image(self, image):
        preview = image.copy()
        preview.thumbnail((650, 450))
        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.photo)

    def upload_image(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if self.file_path:
            image = Image.open(self.file_path).convert('RGB')
            self._show_image(image)
            self.classify_button.config(state=tk.NORMAL)
            self.result_label.config(text="")

    def classify_image(self):
        if not self.file_path:
            return

        predictions = predict_many(self.file_path)
        if not predictions:
            self.result_label.config(text="No signs found or model file is missing.")
            return

        annotated_image = annotate_image(self.file_path, predictions)
        self._show_image(annotated_image)

        lines = ["Detected signs:"]
        for index, prediction in enumerate(predictions, start=1):
            lines.append(
                f"{index}. {prediction['class_name']} "
                f"({prediction['confidence']:.1%})"
            )

        self.result_label.config(text="\n".join(lines))

    def reset(self):
        self.file_path = None
        self.image_label.config(image='')
        self.result_label.config(text="")
        self.classify_button.config(state=tk.DISABLED)
        self.photo = None


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
