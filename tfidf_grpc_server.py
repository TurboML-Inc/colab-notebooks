import grpc
from concurrent import futures
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from turboml.common.protos import (
    input_pb2,
    output_pb2,
    ml_service_pb2_grpc,
    ml_service_pb2,
)

nltk.download("punkt_tab")
tfidf_vectorizer = TfidfVectorizer()


class MLService(ml_service_pb2_grpc.MLServiceServicer):
    def __init__(self) -> None:
        pass

    def Learn(self, request: input_pb2.Input, context) -> ml_service_pb2.Empty:
        try:
            new_corpus = [" ".join(nltk.word_tokenize(text)) for text in request.text]

            if not new_corpus:
                context.set_details("Empty corpus provided")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return ml_service_pb2.Empty()

            tfidf_vectorizer.fit(new_corpus)

            return ml_service_pb2.Empty()

        except Exception as e:
            context.set_details(f"Error during learning: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)

    def Predict(self, request: input_pb2.Input, context) -> output_pb2.Output:
        try:
            embeddings = []
            text_ = request.text[0]
            tokens = nltk.word_tokenize(text_)
            text_input = " ".join(tokens)
            tfidf_embedding = tfidf_vectorizer.transform([text_input]).toarray()
            if tfidf_embedding.size > 0:
                embeddings = tfidf_embedding[0].tolist()
            response = output_pb2.Output(
                key=str(request.key) if request.key else "",
                embeddings=embeddings if embeddings else [],
            )
            return response
        except Exception as e:
            context.set_details(f"Error during learning: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)


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
