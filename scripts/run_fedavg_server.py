import hydra
from flwr_leaf.server.fedavg import start_fedavg_server


@hydra.main(config_path="config", config_name="synthetic_server")
def main(config):
    hist = start_fedavg_server(config.server_address, config.num_rounds)


if __name__ == "__main__":
    main()
