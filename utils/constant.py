from model.config import get_args
opt=get_args()
dst=opt.dst
if dst=='shakespeare' or dst =='sym':
    stopwords = '#$%&()*+--)–/:;<=>@[\\]^_`{|}~—•…�™—' + '0123456789'
elif dst=='yelp' or dst=='amazon':
    stopwords='#$%&()*+--)–/:;<=>@[\\]^_`{|}~—•…�™—'+'0123456789'
elif dst=='gyafc':
    #stopwords = '#$%&()*+,-–./:;<=>@[\\]^_`{|}~•…�'+'0123456789'
    stopwords = '#$%|~…�<=>'

DATASET_DICT ={
    'yelp': {
        'from': ['positive', 'negative'],
        'from_to': {
            'positive': 'negative',
            'negative': 'positive',
            },
        'examples': [
            ('negative', 'this place is awful!'),
            ('positive', 'this place is amazing!', 'negative'),
            ('negative', 'i hated their black tea and hated hot chocolate selections!'),
            ('positive', 'i loved their black tea and loved hot chocolate selections!'),
        ],
    },
    'amazon': {
        'from': ['positive', 'negative'],
        'from_to': {
            'positive': 'negative',
            'negative': 'positive',
            },
        'examples' : [
            ('negative', 'this place is awful!'),
            ('positive', 'this place is amazing!', 'negative'),
            ('negative', 'i hated their black tea and hated hot chocolate selections!'),
            ('positive', 'i loved their black tea and loved hot chocolate selections!'),
        ],
    },
    'gyafc': {
        'from': ['informal', 'formal'],
        'from_to': {
            'informal': 'formal',
            'formal': 'informal',
            },
        'examples' : [
            ('informal', 'ohhh i don\'t intend to be mean ...'),
            ('formal', 'i do not intend to be mean', 'informal'),
            ('informal', ',,, that sucks man but u gotta move on :)'),
            ('formal', 'that is unfortunate, but you need to move on'),

        ]
    },
}


def write_sentence(dataset,setting, delim_left, delim_right, orig_text, rewritten_text=None):

    if dataset=='yelp' or dataset=='amazon':
        style_word='sentiment'
    else: style_word='style'

    if setting =='classification_v1':
        sentence =f'here is a text: {delim_left}{orig_text}{delim_right}. The {style_word} is:'
    elif setting =='classification_v2':
        sentence =f'The {style_word} of the text {delim_left}{orig_text}{delim_right} is: '

    # Basically, if we are creating exemplers, we also have the ground truth (viz., rewritten sentence)
    if rewritten_text is not None:
        sentence = f'{sentence} {rewritten_text}'
    return sentence

# EOSequence token
FS_EOS_TOKEN = '\n###\n'

# Create exemplars (for the few-shot setting)
def create_exemplars(dataset, num_examples, setting, delim_left, delim_right):
    prefix = ''
    exemples=DATASET_DICT[dataset]['examples'][:num_examples]
    for exemple in exemples:
        # ('negative', 'this place is awful!', 'positive', 'this place is amazing!'),
        orig_style, orig_text, opp_style, rewritten_text = exemple
        add_text = write_sentence(dataset,setting, delim_left, delim_right, orig_text, orig_style)
        prefix += f'{add_text}{FS_EOS_TOKEN}'
    return prefix




