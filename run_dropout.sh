THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=1 -beta=0.5
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=1 -beta=-0.5
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=0 -beta=0.5
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=0 -beta=0
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=0 -beta=-0.5
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=-1 -beta=0.5
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=-1 -beta=0
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python dropoutDBN.py \
-drop_type=bernoulli -layer_type=adaptive_dropout -act_type=relu -num_runs=1 -alpha=-1 -beta=-0.5
