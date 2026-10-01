import importlib

DATASET_MODULES = {
    'math500': 'data.math500',
    'formal_logic': 'data.mmlu_formal_logic',
    'deepscaler': 'data.deepscaler',
    'gsm8k': 'data.gsm8k',
    'prog_expr': 'data.prog_expr',
}


def load_data(args, split):
    if args.data not in DATASET_MODULES:
        raise NotImplementedError(f"Unknown dataset: {args.data}")
    return importlib.import_module(DATASET_MODULES[args.data]).load_data(args, split=split)
