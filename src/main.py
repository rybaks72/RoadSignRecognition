import sys


def main():
    print("Road Sign Recognition Application")
    print("---------------------------------")
    print("This application uses a CNN classifier trained on GTSRB.")
    print("It can classify one cropped sign or detect and classify multiple signs in one image.")
    print("")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/main.py train                         - Train the model")
        print("  python src/main.py predict <path> [output_path]  - Detect and classify all signs in image")
        print("  python src/main.py predict-one <path>            - Classify the whole image as one sign")
        print("  python src/main.py gui                           - Launch the graphical user interface")
        return

    command = sys.argv[1].lower()

    if command == "train":
        from train import train_model
        train_model()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("Please provide an image path.")
            return

        from predict import annotate_image, predict_many

        img_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) >= 4 else None
        results = predict_many(img_path)

        if not results:
            print("No signs found or model is missing.")
            return

        print(f"Detected signs: {len(results)}")
        for index, result in enumerate(results, start=1):
            print(
                f"{index}. {result['class_name']} | "
                f"confidence: {result['confidence']:.2%} | "
                f"bbox: {result['bbox']}"
            )

        if output_path:
            annotate_image(img_path, results, output_path)
            print(f"Annotated image saved to: {output_path}")

    elif command == "predict-one":
        if len(sys.argv) < 3:
            print("Please provide an image path.")
            return

        from predict import predict

        img_path = sys.argv[2]
        result = predict(img_path)
        if result:
            print(f"Predicted Traffic Sign: {result}")

    elif command == "gui":
        from gui import main as gui_main
        gui_main()

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
