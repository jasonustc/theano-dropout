THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
-drop_type=uniform -act_type=relu -num_runs=30 -a=0 -b=1
#THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=1 -sigma=0.5 
#THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=1 -sigma=0.6 
#THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=1 -sigma=0.7 
#THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=1 -sigma=0.8 
#THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=1 -sigma=0.9 
#THEANO_FLAGS=device=gpu1,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=1 -sigma=1.0 
