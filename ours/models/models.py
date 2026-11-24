def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'FSFNet':
        assert (opt.dataset_mode == 'unaligned')
        from .FSFNet import FSFNet
        model = FSFNet()
    elif opt.model == 'FSFNetPT':
        assert (opt.dataset_mode == 'unaligned')
        from .FSFNet_pretrain import FSFNetPT
        model = FSFNetPT()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
