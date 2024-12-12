import grpc
from concurrent import futures
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from io import BytesIO

from turboml.common.protos import (
    input_pb2,
    output_pb2,
    ml_service_pb2_grpc,
    ml_service_pb2,
)


def get_classes(path: str) -> list:
    with open(path) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


class MLService(ml_service_pb2_grpc.MLServiceServicer):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = get_classes("data/imagenet_labels.txt")
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(
            self.device
        )
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def Learn(self, request: input_pb2.Input, context) -> ml_service_pb2.Empty:
        return ml_service_pb2.Empty()

    def Predict(self, request: input_pb2.Input, context) -> output_pb2.Output:
        if not request.images:
            context.set_details("No images provided.")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return output_pb2.Output()

        try:
            image_data = request.images[0]
            img = Image.open(BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            predicted_index = torch.argmax(probabilities)

            response = output_pb2.Output(
                key=request.key,
                text_output=f"{self.classes[predicted_index]},{probabilities[predicted_index]}",
            )

            return response

        except Exception as e:
            context.set_details(f"Error during prediction: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return output_pb2.Output()


def serve(server_url: str) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    try:
        ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLService(), server)
        server.add_insecure_port(server_url)
        print(f"Server starting on: {server_url}")
        server.start()
        print(f"Server started successfully on: {server_url}")

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Shutting down server.")
        finally:
            print("Stopping server...")
            server.stop(0)
            print("Server stopped.")

    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        if server:
            server.stop(0)
        raise
