def load_data(args, split):
    if args.data == 'arithmetics' :
        from data.arithmetics import load_data as load_arithmetics
        return load_arithmetics(args, split=split)
    elif args.data == 'hellaswag' :
        from data.hellaswag import load_data as load_hellaswag
        return load_hellaswag(args, split=split)
    elif args.data == 'pro_medicine' :
        from data.mmlu_pro_medicine import load_data 
        return load_data(args, split=split)
    elif args.data == 'formal_logic' :
        from data.mmlu_formal_logic import load_data 
        return load_data(args, split=split)
    elif args.data == 'gsm8k' :
        from data.gsm8k import load_data as load_gsm8k
        return load_gsm8k(args, split=split)
    elif args.data == 'csqa' :
        from data.csqa import load_data as load_csqa
        return load_csqa(args, split=split)
    elif args.data == 'hh_rlhf':
        from data.hh_rlhf import load_data as load_hhrlhf
        return load_hhrlhf(args, split=split)
    
    elif args.data == 'deepscaler':
        from data.deepscaler import load_data as load_deepscaler
        return load_deepscaler(args, split=split)
    
    elif args.data == 'mmlu_pro':
        from data.mmlu_pro import load_data
        return load_data(args, split=split)

    elif args.data == 'aime24':
        from data.aime24 import load_data as load_aime24
        return load_aime24(args, split=split)

    elif args.data == 'aime25':
        from data.aime25 import load_data as load_aime25
        return load_aime25(args, split=split)

    elif args.data == 'math500':
        from data.math500 import load_data as load_math500
        return load_math500(args, split=split)

    elif args.data == 'amc23':
        from data.amc23 import load_data as load_amc23
        return load_amc23(args, split=split)

    elif args.data == 'hle':
        from data.hle import load_data as load_hle
        return load_hle(args, split=split)

    elif args.data == 'minerva':
        from data.minerva_math import load_data as load_minerva_math
        return load_minerva_math(args, split=split)

    elif args.data == 'gpqa':
        from data.gpqa import load_gpqa
        return load_gpqa(args, split=split)

    elif args.data == 'math_algebra':
        from data.math_algebra import load_data as load_math_algebra
        return load_math_algebra(args, split=split)