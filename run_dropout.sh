#THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=5 -sigma=0.1 --noclip 
#THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=5 -sigma=0.2 --noclip 
#THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=5 -sigma=0.3 --noclip 
#THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=5 -sigma=0.1 
THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
-drop_type=gaussian -act_type=relu -num_runs=30 -sigma=0.2 
#THEANO_FLAGS=device=gpu0,floatX=float32,mode=FAST_RUN python2.6 dropoutDBN.py \
#-drop_type=gaussian -act_type=relu -num_runs=5 -sigma=0.3 
