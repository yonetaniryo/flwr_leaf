import flwr as fl
import hydra
from flwr_leaf.client.client import Client


@hydra.main(config_path="config", config_name="synthetic_client")
def main(config):
    fl.client.start_numpy_client(
        server_address=config.client_address, client=Client(config)
    )


if __name__ == "__main__":
    main()
