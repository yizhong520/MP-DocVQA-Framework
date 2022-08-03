import os


def save_model(model, epoch, update_best=False, **kwargs):
    save_dir = os.path.join(kwargs['save_dir'], 'checkpoints', "{:s}_{:s}_{:s}".format(kwargs['model_name'].lower(), kwargs['page_retrieval'].lower(), kwargs['dataset_name'].lower()))
    model.model.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))
    model.tokenizer.save_pretrained(os.path.join(save_dir, "model__{:d}.ckpt".format(epoch)))

    if update_best:
        model.model.save_pretrained(os.path.join(save_dir, "best.ckpt"))
        model.tokenizer.save_pretrained(os.path.join(save_dir, "best.ckpt"))


def load_model(base_model, ckpt_name, **kwargs):
    load_dir = kwargs['save_dir']
    base_model.model.from_pretrained(os.path.join(load_dir, ckpt_name))
