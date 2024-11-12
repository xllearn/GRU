import torch
from config import Config
from d2l import torch as d2l
from model import  get_params, init_gru_state, gru

from d2l import torch as d2l
def main():
    device = d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(Config.batch_size, Config.num_steps)
    model = d2l.RNNModelScratch(len(vocab), Config.num_hiddens, device, get_params, init_gru_state, gru)
    d2l.train_ch8(model, train_iter, vocab, Config.lr, Config.num_epochs, device)
    d2l.plt.show()


if __name__ == "__main__":
    main()
