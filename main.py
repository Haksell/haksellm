import torch


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    t0 = torch.tensor(1)
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t3 = torch.tensor([[[1, 2], [3.14, 4]], [[5, 6], [7, 8]]])
    print(t0.shape, t0.dtype)
    print(t1.shape, t1.dtype)
    print(t2.shape, t2.dtype)
    print(t3.shape, t3.dtype)
    t3f64 = t3.to(torch.float64)
    print(t3f64.shape, t3f64.dtype)
    print(t2.view((3, 2)))
    print(t2.T)


if __name__ == "__main__":
    main()
