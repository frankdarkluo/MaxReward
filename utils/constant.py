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
