æ
Ô
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68åÇ

conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv2d_62/kernel
}
$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*&
_output_shapes
:
*
dtype0
t
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_62/bias
m
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes
:
*
dtype0

conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_63/kernel
}
$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_63/bias
m
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes
:
*
dtype0
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¨¬* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
¨¬*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¾
3sequential_64/sequential_63/random_flip_32/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*D
shared_name53sequential_64/sequential_63/random_flip_32/StateVar
·
Gsequential_64/sequential_63/random_flip_32/StateVar/Read/ReadVariableOpReadVariableOp3sequential_64/sequential_63/random_flip_32/StateVar*
_output_shapes
:*
dtype0	
Æ
7sequential_64/sequential_63/random_rotation_29/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*H
shared_name97sequential_64/sequential_63/random_rotation_29/StateVar
¿
Ksequential_64/sequential_63/random_rotation_29/StateVar/Read/ReadVariableOpReadVariableOp7sequential_64/sequential_63/random_rotation_29/StateVar*
_output_shapes
:*
dtype0	
¾
3sequential_64/sequential_63/random_zoom_32/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*D
shared_name53sequential_64/sequential_63/random_zoom_32/StateVar
·
Gsequential_64/sequential_63/random_zoom_32/StateVar/Read/ReadVariableOpReadVariableOp3sequential_64/sequential_63/random_zoom_32/StateVar*
_output_shapes
:*
dtype0	

Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_62/kernel/m

+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*&
_output_shapes
:
*
dtype0

Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_62/bias/m
{
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_63/kernel/m

+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*&
_output_shapes
:

*
dtype0

Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_63/bias/m
{
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes
:
*
dtype0

Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¨¬*'
shared_nameAdam/dense_31/kernel/m

*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
¨¬*
dtype0

Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/m
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_62/kernel/v

+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*&
_output_shapes
:
*
dtype0

Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_62/bias/v
{
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes
:
*
dtype0

Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_63/kernel/v

+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*&
_output_shapes
:

*
dtype0

Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_63/bias/v
{
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes
:
*
dtype0

Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¨¬*'
shared_nameAdam/dense_31/kernel/v

*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
¨¬*
dtype0

Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/v
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ã]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*]
value]B] B]
ª
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-2
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
·
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses*
¥
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*_random_generator
+__call__
*,&call_and_return_all_conditional_losses* 

-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
¥
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses* 
¦

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses*
¥
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F_random_generator
G__call__
*H&call_and_return_all_conditional_losses* 

I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
¥
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S_random_generator
T__call__
*U&call_and_return_all_conditional_losses* 

V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses* 
¦

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
¼
diter

ebeta_1

fbeta_2
	gdecay
hlearning_ratemÛmÜ:mÝ;mÞ\mß]màvávâ:vã;vä\vå]væ*
.
0
1
:2
;3
\4
]5*
.
0
1
:2
;3
\4
]5*
* 
°
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

nserving_default* 
§
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses*
§
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z_random_generator
{__call__
*|&call_and_return_all_conditional_losses*
«
}	variables
~trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEconv2d_62/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_62/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 
* 
* 
* 
`Z
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

:0
;1*

:0
;1*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_31/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

\0
]1*

\0
]1*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

»0
¼1*
* 
* 
* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

Â
_generator*
* 
* 
* 
* 
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

È
_generator*
* 
* 
* 
* 
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Î
_generator*
* 
* 
* 

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

Ïtotal

Ðcount
Ñ	variables
Ò	keras_api*
M

Ótotal

Ôcount
Õ
_fn_kwargs
Ö	variables
×	keras_api*
* 
* 
* 
* 
* 

Ø
_state_var*
* 
* 
* 
* 
* 

Ù
_state_var*
* 
* 
* 
* 
* 

Ú
_state_var*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Ï0
Ð1*

Ñ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ó0
Ô1*

Ö	variables*
 
VARIABLE_VALUE3sequential_64/sequential_63/random_flip_32/StateVarRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
¤
VARIABLE_VALUE7sequential_64/sequential_63/random_rotation_29/StateVarRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
 
VARIABLE_VALUE3sequential_64/sequential_63/random_zoom_32/StateVarRlayer-0/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_62/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_62/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_62/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_62/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_sequential_63_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
°
StatefulPartitionedCallStatefulPartitionedCall#serving_default_sequential_63_inputconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasdense_31/kerneldense_31/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_76337
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpGsequential_64/sequential_63/random_flip_32/StateVar/Read/ReadVariableOpKsequential_64/sequential_63/random_rotation_29/StateVar/Read/ReadVariableOpGsequential_64/sequential_63/random_zoom_32/StateVar/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOpConst*+
Tin$
"2 				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_77386
¾
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasdense_31/kerneldense_31/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_13sequential_64/sequential_63/random_flip_32/StateVar7sequential_64/sequential_63/random_rotation_29/StateVar3sequential_64/sequential_63/random_zoom_32/StateVarAdam/conv2d_62/kernel/mAdam/conv2d_62/bias/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/v**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_77486ê
í

)__inference_conv2d_63_layer_call_fn_76773

inputs!
unknown:


	unknown_0:

identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_75513w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

d
+__inference_dropout_119_layer_call_fn_76831

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_119_layer_call_and_return_conditional_losses_75611w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs
¶
F
*__inference_flatten_28_layer_call_fn_76853

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_75540b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs

d
+__inference_dropout_118_layer_call_fn_76794

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_118_layer_call_and_return_conditional_losses_75634w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

 
_user_specified_nameinputs
ù
d
F__inference_dropout_119_layer_call_and_return_conditional_losses_76836

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs
Ð
I
-__inference_sequential_63_layer_call_fn_76342

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75005j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

(__inference_dense_31_layer_call_fn_76868

inputs
unknown:
¨¬
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_75553o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬
 
_user_specified_nameinputs
ê
Ã	
H__inference_sequential_64_layer_call_and_return_conditional_losses_76318

inputs\
Nsequential_63_random_flip_32_stateful_uniform_full_int_rngreadandskip_resource:	W
Isequential_63_random_rotation_29_stateful_uniform_rngreadandskip_resource:	S
Esequential_63_random_zoom_32_stateful_uniform_rngreadandskip_resource:	B
(conv2d_62_conv2d_readvariableop_resource:
7
)conv2d_62_biasadd_readvariableop_resource:
B
(conv2d_63_conv2d_readvariableop_resource:

7
)conv2d_63_biasadd_readvariableop_resource:
;
'dense_31_matmul_readvariableop_resource:
¨¬6
(dense_31_biasadd_readvariableop_resource:
identity¢ conv2d_62/BiasAdd/ReadVariableOp¢conv2d_62/Conv2D/ReadVariableOp¢ conv2d_63/BiasAdd/ReadVariableOp¢conv2d_63/Conv2D/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢Esequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkip¢Gsequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip¢@sequential_63/random_rotation_29/stateful_uniform/RngReadAndSkip¢<sequential_63/random_zoom_32/stateful_uniform/RngReadAndSkip
<sequential_63/random_flip_32/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
<sequential_63/random_flip_32/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ò
;sequential_63/random_flip_32/stateful_uniform_full_int/ProdProdEsequential_63/random_flip_32/stateful_uniform_full_int/shape:output:0Esequential_63/random_flip_32/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 
=sequential_63/random_flip_32/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :»
=sequential_63/random_flip_32/stateful_uniform_full_int/Cast_1CastDsequential_63/random_flip_32/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Î
Esequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipNsequential_63_random_flip_32_stateful_uniform_full_int_rngreadandskip_resourceFsequential_63/random_flip_32/stateful_uniform_full_int/Cast/x:output:0Asequential_63/random_flip_32/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Jsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
Dsequential_63/random_flip_32/stateful_uniform_full_int/strided_sliceStridedSliceMsequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkip:value:0Ssequential_63/random_flip_32/stateful_uniform_full_int/strided_slice/stack:output:0Usequential_63/random_flip_32/stateful_uniform_full_int/strided_slice/stack_1:output:0Usequential_63/random_flip_32/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÉ
>sequential_63/random_flip_32/stateful_uniform_full_int/BitcastBitcastMsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Lsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Nsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
Fsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1StridedSliceMsequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkip:value:0Usequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1/stack:output:0Wsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Wsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Í
@sequential_63/random_flip_32/stateful_uniform_full_int/Bitcast_1BitcastOsequential_63/random_flip_32/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0|
:sequential_63/random_flip_32/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :¢
6sequential_63/random_flip_32/stateful_uniform_full_intStatelessRandomUniformFullIntV2Esequential_63/random_flip_32/stateful_uniform_full_int/shape:output:0Isequential_63/random_flip_32/stateful_uniform_full_int/Bitcast_1:output:0Gsequential_63/random_flip_32/stateful_uniform_full_int/Bitcast:output:0Csequential_63/random_flip_32/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	q
'sequential_63/random_flip_32/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Ï
"sequential_63/random_flip_32/stackPack?sequential_63/random_flip_32/stateful_uniform_full_int:output:00sequential_63/random_flip_32/zeros_like:output:0*
N*
T0	*
_output_shapes

:
0sequential_63/random_flip_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
2sequential_63/random_flip_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2sequential_63/random_flip_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
*sequential_63/random_flip_32/strided_sliceStridedSlice+sequential_63/random_flip_32/stack:output:09sequential_63/random_flip_32/strided_slice/stack:output:0;sequential_63/random_flip_32/strided_slice/stack_1:output:0;sequential_63/random_flip_32/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask»
Psequential_63/random_flip_32/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
Csequential_63/random_flip_32/stateless_random_flip_left_right/ShapeShapeYsequential_63/random_flip_32/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Qsequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ssequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ssequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_63/random_flip_32/stateless_random_flip_left_right/strided_sliceStridedSliceLsequential_63/random_flip_32/stateless_random_flip_left_right/Shape:output:0Zsequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice/stack:output:0\sequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice/stack_1:output:0\sequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskè
\sequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/shapePackTsequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Zsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Zsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ê
ssequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter3sequential_63/random_flip_32/strided_slice:output:0* 
_output_shapes
::µ
ssequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
osequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2esequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0ysequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0}sequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0|sequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
Zsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/subSubcsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/max:output:0csequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: é
Zsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/mulMulxsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0^sequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
Vsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniformAddV2^sequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0csequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Msequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Msequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Msequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ß
Ksequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shapePackTsequential_63/random_flip_32/stateless_random_flip_left_right/strided_slice:output:0Vsequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape/1:output:0Vsequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape/2:output:0Vsequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¼
Esequential_63/random_flip_32/stateless_random_flip_left_right/ReshapeReshapeZsequential_63/random_flip_32/stateless_random_flip_left_right/stateless_random_uniform:z:0Tsequential_63/random_flip_32/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
Csequential_63/random_flip_32/stateless_random_flip_left_right/RoundRoundNsequential_63/random_flip_32/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_63/random_flip_32/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:Â
Gsequential_63/random_flip_32/stateless_random_flip_left_right/ReverseV2	ReverseV2Ysequential_63/random_flip_32/stateless_random_flip_left_right/control_dependency:output:0Usequential_63/random_flip_32/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Asequential_63/random_flip_32/stateless_random_flip_left_right/mulMulGsequential_63/random_flip_32/stateless_random_flip_left_right/Round:y:0Psequential_63/random_flip_32/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Csequential_63/random_flip_32/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Asequential_63/random_flip_32/stateless_random_flip_left_right/subSubLsequential_63/random_flip_32/stateless_random_flip_left_right/sub/x:output:0Gsequential_63/random_flip_32/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
Csequential_63/random_flip_32/stateless_random_flip_left_right/mul_1MulEsequential_63/random_flip_32/stateless_random_flip_left_right/sub:z:0Ysequential_63/random_flip_32/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Asequential_63/random_flip_32/stateless_random_flip_left_right/addAddV2Esequential_63/random_flip_32/stateless_random_flip_left_right/mul:z:0Gsequential_63/random_flip_32/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_63/random_flip_32/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
>sequential_63/random_flip_32/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ø
=sequential_63/random_flip_32/stateful_uniform_full_int_1/ProdProdGsequential_63/random_flip_32/stateful_uniform_full_int_1/shape:output:0Gsequential_63/random_flip_32/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 
?sequential_63/random_flip_32/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¿
?sequential_63/random_flip_32/stateful_uniform_full_int_1/Cast_1CastFsequential_63/random_flip_32/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
Gsequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkipNsequential_63_random_flip_32_stateful_uniform_full_int_rngreadandskip_resourceHsequential_63/random_flip_32/stateful_uniform_full_int_1/Cast/x:output:0Csequential_63/random_flip_32/stateful_uniform_full_int_1/Cast_1:y:0F^sequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:
Lsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
Fsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_sliceStridedSliceOsequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip:value:0Usequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice/stack:output:0Wsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Wsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÍ
@sequential_63/random_flip_32/stateful_uniform_full_int_1/BitcastBitcastOsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Nsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Psequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Psequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
Hsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1StridedSliceOsequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip:value:0Wsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Ysequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Ysequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ñ
Bsequential_63/random_flip_32/stateful_uniform_full_int_1/Bitcast_1BitcastQsequential_63/random_flip_32/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0~
<sequential_63/random_flip_32/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :¬
8sequential_63/random_flip_32/stateful_uniform_full_int_1StatelessRandomUniformFullIntV2Gsequential_63/random_flip_32/stateful_uniform_full_int_1/shape:output:0Ksequential_63/random_flip_32/stateful_uniform_full_int_1/Bitcast_1:output:0Isequential_63/random_flip_32/stateful_uniform_full_int_1/Bitcast:output:0Esequential_63/random_flip_32/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	s
)sequential_63/random_flip_32/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R Õ
$sequential_63/random_flip_32/stack_1PackAsequential_63/random_flip_32/stateful_uniform_full_int_1:output:02sequential_63/random_flip_32/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:
2sequential_63/random_flip_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
4sequential_63/random_flip_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
4sequential_63/random_flip_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
,sequential_63/random_flip_32/strided_slice_1StridedSlice-sequential_63/random_flip_32/stack_1:output:0;sequential_63/random_flip_32/strided_slice_1/stack:output:0=sequential_63/random_flip_32/strided_slice_1/stack_1:output:0=sequential_63/random_flip_32/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask²
Msequential_63/random_flip_32/stateless_random_flip_up_down/control_dependencyIdentityEsequential_63/random_flip_32/stateless_random_flip_left_right/add:z:0*
T0*T
_classJ
HFloc:@sequential_63/random_flip_32/stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
@sequential_63/random_flip_32/stateless_random_flip_up_down/ShapeShapeVsequential_63/random_flip_32/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:
Nsequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Psequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Psequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
Hsequential_63/random_flip_32/stateless_random_flip_up_down/strided_sliceStridedSliceIsequential_63/random_flip_32/stateless_random_flip_up_down/Shape:output:0Wsequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice/stack:output:0Ysequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice/stack_1:output:0Ysequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskâ
Ysequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/shapePackQsequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
Wsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Wsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?é
psequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter5sequential_63/random_flip_32/strided_slice_1:output:0* 
_output_shapes
::²
psequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
lsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2bsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0vsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0zsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0ysequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
Wsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/subSub`sequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/max:output:0`sequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: à
Wsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/mulMulusequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0[sequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Ssequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniformAddV2[sequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0`sequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Jsequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Jsequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Jsequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ð
Hsequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shapePackQsequential_63/random_flip_32/stateless_random_flip_up_down/strided_slice:output:0Ssequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape/1:output:0Ssequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape/2:output:0Ssequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:³
Bsequential_63/random_flip_32/stateless_random_flip_up_down/ReshapeReshapeWsequential_63/random_flip_32/stateless_random_flip_up_down/stateless_random_uniform:z:0Qsequential_63/random_flip_32/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
@sequential_63/random_flip_32/stateless_random_flip_up_down/RoundRoundKsequential_63/random_flip_32/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Isequential_63/random_flip_32/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:¹
Dsequential_63/random_flip_32/stateless_random_flip_up_down/ReverseV2	ReverseV2Vsequential_63/random_flip_32/stateless_random_flip_up_down/control_dependency:output:0Rsequential_63/random_flip_32/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_63/random_flip_32/stateless_random_flip_up_down/mulMulDsequential_63/random_flip_32/stateless_random_flip_up_down/Round:y:0Msequential_63/random_flip_32/stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential_63/random_flip_32/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>sequential_63/random_flip_32/stateless_random_flip_up_down/subSubIsequential_63/random_flip_32/stateless_random_flip_up_down/sub/x:output:0Dsequential_63/random_flip_32/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential_63/random_flip_32/stateless_random_flip_up_down/mul_1MulBsequential_63/random_flip_32/stateless_random_flip_up_down/sub:z:0Vsequential_63/random_flip_32/stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_63/random_flip_32/stateless_random_flip_up_down/addAddV2Bsequential_63/random_flip_32/stateless_random_flip_up_down/mul:z:0Dsequential_63/random_flip_32/stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_63/random_rotation_29/ShapeShapeBsequential_63/random_flip_32/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:~
4sequential_63/random_rotation_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6sequential_63/random_rotation_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_63/random_rotation_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.sequential_63/random_rotation_29/strided_sliceStridedSlice/sequential_63/random_rotation_29/Shape:output:0=sequential_63/random_rotation_29/strided_slice/stack:output:0?sequential_63/random_rotation_29/strided_slice/stack_1:output:0?sequential_63/random_rotation_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6sequential_63/random_rotation_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
8sequential_63/random_rotation_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
8sequential_63/random_rotation_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0sequential_63/random_rotation_29/strided_slice_1StridedSlice/sequential_63/random_rotation_29/Shape:output:0?sequential_63/random_rotation_29/strided_slice_1/stack:output:0Asequential_63/random_rotation_29/strided_slice_1/stack_1:output:0Asequential_63/random_rotation_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
%sequential_63/random_rotation_29/CastCast9sequential_63/random_rotation_29/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
6sequential_63/random_rotation_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
8sequential_63/random_rotation_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
8sequential_63/random_rotation_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0sequential_63/random_rotation_29/strided_slice_2StridedSlice/sequential_63/random_rotation_29/Shape:output:0?sequential_63/random_rotation_29/strided_slice_2/stack:output:0Asequential_63/random_rotation_29/strided_slice_2/stack_1:output:0Asequential_63/random_rotation_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'sequential_63/random_rotation_29/Cast_1Cast9sequential_63/random_rotation_29/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: ¦
7sequential_63/random_rotation_29/stateful_uniform/shapePack7sequential_63/random_rotation_29/strided_slice:output:0*
N*
T0*
_output_shapes
:z
5sequential_63/random_rotation_29/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿z
5sequential_63/random_rotation_29/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?
7sequential_63/random_rotation_29/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ã
6sequential_63/random_rotation_29/stateful_uniform/ProdProd@sequential_63/random_rotation_29/stateful_uniform/shape:output:0@sequential_63/random_rotation_29/stateful_uniform/Const:output:0*
T0*
_output_shapes
: z
8sequential_63/random_rotation_29/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :±
8sequential_63/random_rotation_29/stateful_uniform/Cast_1Cast?sequential_63/random_rotation_29/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: º
@sequential_63/random_rotation_29/stateful_uniform/RngReadAndSkipRngReadAndSkipIsequential_63_random_rotation_29_stateful_uniform_rngreadandskip_resourceAsequential_63/random_rotation_29/stateful_uniform/Cast/x:output:0<sequential_63/random_rotation_29/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Esequential_63/random_rotation_29/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential_63/random_rotation_29/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gsequential_63/random_rotation_29/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
?sequential_63/random_rotation_29/stateful_uniform/strided_sliceStridedSliceHsequential_63/random_rotation_29/stateful_uniform/RngReadAndSkip:value:0Nsequential_63/random_rotation_29/stateful_uniform/strided_slice/stack:output:0Psequential_63/random_rotation_29/stateful_uniform/strided_slice/stack_1:output:0Psequential_63/random_rotation_29/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask¿
9sequential_63/random_rotation_29/stateful_uniform/BitcastBitcastHsequential_63/random_rotation_29/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Gsequential_63/random_rotation_29/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Isequential_63/random_rotation_29/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Isequential_63/random_rotation_29/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
Asequential_63/random_rotation_29/stateful_uniform/strided_slice_1StridedSliceHsequential_63/random_rotation_29/stateful_uniform/RngReadAndSkip:value:0Psequential_63/random_rotation_29/stateful_uniform/strided_slice_1/stack:output:0Rsequential_63/random_rotation_29/stateful_uniform/strided_slice_1/stack_1:output:0Rsequential_63/random_rotation_29/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ã
;sequential_63/random_rotation_29/stateful_uniform/Bitcast_1BitcastJsequential_63/random_rotation_29/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Nsequential_63/random_rotation_29/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :°
Jsequential_63/random_rotation_29/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2@sequential_63/random_rotation_29/stateful_uniform/shape:output:0Dsequential_63/random_rotation_29/stateful_uniform/Bitcast_1:output:0Bsequential_63/random_rotation_29/stateful_uniform/Bitcast:output:0Wsequential_63/random_rotation_29/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
5sequential_63/random_rotation_29/stateful_uniform/subSub>sequential_63/random_rotation_29/stateful_uniform/max:output:0>sequential_63/random_rotation_29/stateful_uniform/min:output:0*
T0*
_output_shapes
: ú
5sequential_63/random_rotation_29/stateful_uniform/mulMulSsequential_63/random_rotation_29/stateful_uniform/StatelessRandomUniformV2:output:09sequential_63/random_rotation_29/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
1sequential_63/random_rotation_29/stateful_uniformAddV29sequential_63/random_rotation_29/stateful_uniform/mul:z:0>sequential_63/random_rotation_29/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
6sequential_63/random_rotation_29/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ê
4sequential_63/random_rotation_29/rotation_matrix/subSub+sequential_63/random_rotation_29/Cast_1:y:0?sequential_63/random_rotation_29/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
:  
4sequential_63/random_rotation_29/rotation_matrix/CosCos5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8sequential_63/random_rotation_29/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Î
6sequential_63/random_rotation_29/rotation_matrix/sub_1Sub+sequential_63/random_rotation_29/Cast_1:y:0Asequential_63/random_rotation_29/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: ß
4sequential_63/random_rotation_29/rotation_matrix/mulMul8sequential_63/random_rotation_29/rotation_matrix/Cos:y:0:sequential_63/random_rotation_29/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4sequential_63/random_rotation_29/rotation_matrix/SinSin5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8sequential_63/random_rotation_29/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ì
6sequential_63/random_rotation_29/rotation_matrix/sub_2Sub)sequential_63/random_rotation_29/Cast:y:0Asequential_63/random_rotation_29/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: á
6sequential_63/random_rotation_29/rotation_matrix/mul_1Mul8sequential_63/random_rotation_29/rotation_matrix/Sin:y:0:sequential_63/random_rotation_29/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
6sequential_63/random_rotation_29/rotation_matrix/sub_3Sub8sequential_63/random_rotation_29/rotation_matrix/mul:z:0:sequential_63/random_rotation_29/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
6sequential_63/random_rotation_29/rotation_matrix/sub_4Sub8sequential_63/random_rotation_29/rotation_matrix/sub:z:0:sequential_63/random_rotation_29/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:sequential_63/random_rotation_29/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ò
8sequential_63/random_rotation_29/rotation_matrix/truedivRealDiv:sequential_63/random_rotation_29/rotation_matrix/sub_4:z:0Csequential_63/random_rotation_29/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8sequential_63/random_rotation_29/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ì
6sequential_63/random_rotation_29/rotation_matrix/sub_5Sub)sequential_63/random_rotation_29/Cast:y:0Asequential_63/random_rotation_29/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: ¢
6sequential_63/random_rotation_29/rotation_matrix/Sin_1Sin5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8sequential_63/random_rotation_29/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Î
6sequential_63/random_rotation_29/rotation_matrix/sub_6Sub+sequential_63/random_rotation_29/Cast_1:y:0Asequential_63/random_rotation_29/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ã
6sequential_63/random_rotation_29/rotation_matrix/mul_2Mul:sequential_63/random_rotation_29/rotation_matrix/Sin_1:y:0:sequential_63/random_rotation_29/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
6sequential_63/random_rotation_29/rotation_matrix/Cos_1Cos5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
8sequential_63/random_rotation_29/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ì
6sequential_63/random_rotation_29/rotation_matrix/sub_7Sub)sequential_63/random_rotation_29/Cast:y:0Asequential_63/random_rotation_29/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ã
6sequential_63/random_rotation_29/rotation_matrix/mul_3Mul:sequential_63/random_rotation_29/rotation_matrix/Cos_1:y:0:sequential_63/random_rotation_29/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
4sequential_63/random_rotation_29/rotation_matrix/addAddV2:sequential_63/random_rotation_29/rotation_matrix/mul_2:z:0:sequential_63/random_rotation_29/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
6sequential_63/random_rotation_29/rotation_matrix/sub_8Sub:sequential_63/random_rotation_29/rotation_matrix/sub_5:z:08sequential_63/random_rotation_29/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<sequential_63/random_rotation_29/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ö
:sequential_63/random_rotation_29/rotation_matrix/truediv_1RealDiv:sequential_63/random_rotation_29/rotation_matrix/sub_8:z:0Esequential_63/random_rotation_29/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6sequential_63/random_rotation_29/rotation_matrix/ShapeShape5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*
_output_shapes
:
Dsequential_63/random_rotation_29/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Æ
>sequential_63/random_rotation_29/rotation_matrix/strided_sliceStridedSlice?sequential_63/random_rotation_29/rotation_matrix/Shape:output:0Msequential_63/random_rotation_29/rotation_matrix/strided_slice/stack:output:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice/stack_1:output:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¢
6sequential_63/random_rotation_29/rotation_matrix/Cos_2Cos5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ù
@sequential_63/random_rotation_29/rotation_matrix/strided_slice_1StridedSlice:sequential_63/random_rotation_29/rotation_matrix/Cos_2:y:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice_1/stack:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_1/stack_1:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¢
6sequential_63/random_rotation_29/rotation_matrix/Sin_2Sin5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ù
@sequential_63/random_rotation_29/rotation_matrix/strided_slice_2StridedSlice:sequential_63/random_rotation_29/rotation_matrix/Sin_2:y:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice_2/stack:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_2/stack_1:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¸
4sequential_63/random_rotation_29/rotation_matrix/NegNegIsequential_63/random_rotation_29/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      û
@sequential_63/random_rotation_29/rotation_matrix/strided_slice_3StridedSlice<sequential_63/random_rotation_29/rotation_matrix/truediv:z:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice_3/stack:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_3/stack_1:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¢
6sequential_63/random_rotation_29/rotation_matrix/Sin_3Sin5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ù
@sequential_63/random_rotation_29/rotation_matrix/strided_slice_4StridedSlice:sequential_63/random_rotation_29/rotation_matrix/Sin_3:y:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice_4/stack:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_4/stack_1:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¢
6sequential_63/random_rotation_29/rotation_matrix/Cos_3Cos5sequential_63/random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ù
@sequential_63/random_rotation_29/rotation_matrix/strided_slice_5StridedSlice:sequential_63/random_rotation_29/rotation_matrix/Cos_3:y:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice_5/stack:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_5/stack_1:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
Fsequential_63/random_rotation_29/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Hsequential_63/random_rotation_29/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
@sequential_63/random_rotation_29/rotation_matrix/strided_slice_6StridedSlice>sequential_63/random_rotation_29/rotation_matrix/truediv_1:z:0Osequential_63/random_rotation_29/rotation_matrix/strided_slice_6/stack:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_6/stack_1:output:0Qsequential_63/random_rotation_29/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
?sequential_63/random_rotation_29/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
=sequential_63/random_rotation_29/rotation_matrix/zeros/packedPackGsequential_63/random_rotation_29/rotation_matrix/strided_slice:output:0Hsequential_63/random_rotation_29/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
<sequential_63/random_rotation_29/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ÿ
6sequential_63/random_rotation_29/rotation_matrix/zerosFillFsequential_63/random_rotation_29/rotation_matrix/zeros/packed:output:0Esequential_63/random_rotation_29/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
<sequential_63/random_rotation_29/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :·
7sequential_63/random_rotation_29/rotation_matrix/concatConcatV2Isequential_63/random_rotation_29/rotation_matrix/strided_slice_1:output:08sequential_63/random_rotation_29/rotation_matrix/Neg:y:0Isequential_63/random_rotation_29/rotation_matrix/strided_slice_3:output:0Isequential_63/random_rotation_29/rotation_matrix/strided_slice_4:output:0Isequential_63/random_rotation_29/rotation_matrix/strided_slice_5:output:0Isequential_63/random_rotation_29/rotation_matrix/strided_slice_6:output:0?sequential_63/random_rotation_29/rotation_matrix/zeros:output:0Esequential_63/random_rotation_29/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
0sequential_63/random_rotation_29/transform/ShapeShapeBsequential_63/random_flip_32/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:
>sequential_63/random_rotation_29/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
@sequential_63/random_rotation_29/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@sequential_63/random_rotation_29/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
8sequential_63/random_rotation_29/transform/strided_sliceStridedSlice9sequential_63/random_rotation_29/transform/Shape:output:0Gsequential_63/random_rotation_29/transform/strided_slice/stack:output:0Isequential_63/random_rotation_29/transform/strided_slice/stack_1:output:0Isequential_63/random_rotation_29/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:z
5sequential_63/random_rotation_29/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    á
Esequential_63/random_rotation_29/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Bsequential_63/random_flip_32/stateless_random_flip_up_down/add:z:0@sequential_63/random_rotation_29/rotation_matrix/concat:output:0Asequential_63/random_rotation_29/transform/strided_slice:output:0>sequential_63/random_rotation_29/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR¬
"sequential_63/random_zoom_32/ShapeShapeZsequential_63/random_rotation_29/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:z
0sequential_63/random_zoom_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_63/random_zoom_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_63/random_zoom_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*sequential_63/random_zoom_32/strided_sliceStridedSlice+sequential_63/random_zoom_32/Shape:output:09sequential_63/random_zoom_32/strided_slice/stack:output:0;sequential_63/random_zoom_32/strided_slice/stack_1:output:0;sequential_63/random_zoom_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
2sequential_63/random_zoom_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
4sequential_63/random_zoom_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ~
4sequential_63/random_zoom_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
,sequential_63/random_zoom_32/strided_slice_1StridedSlice+sequential_63/random_zoom_32/Shape:output:0;sequential_63/random_zoom_32/strided_slice_1/stack:output:0=sequential_63/random_zoom_32/strided_slice_1/stack_1:output:0=sequential_63/random_zoom_32/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!sequential_63/random_zoom_32/CastCast5sequential_63/random_zoom_32/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2sequential_63/random_zoom_32/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
4sequential_63/random_zoom_32/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ~
4sequential_63/random_zoom_32/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ê
,sequential_63/random_zoom_32/strided_slice_2StridedSlice+sequential_63/random_zoom_32/Shape:output:0;sequential_63/random_zoom_32/strided_slice_2/stack:output:0=sequential_63/random_zoom_32/strided_slice_2/stack_1:output:0=sequential_63/random_zoom_32/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#sequential_63/random_zoom_32/Cast_1Cast5sequential_63/random_zoom_32/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: w
5sequential_63/random_zoom_32/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Þ
3sequential_63/random_zoom_32/stateful_uniform/shapePack3sequential_63/random_zoom_32/strided_slice:output:0>sequential_63/random_zoom_32/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:v
1sequential_63/random_zoom_32/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?v
1sequential_63/random_zoom_32/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?}
3sequential_63/random_zoom_32/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ×
2sequential_63/random_zoom_32/stateful_uniform/ProdProd<sequential_63/random_zoom_32/stateful_uniform/shape:output:0<sequential_63/random_zoom_32/stateful_uniform/Const:output:0*
T0*
_output_shapes
: v
4sequential_63/random_zoom_32/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :©
4sequential_63/random_zoom_32/stateful_uniform/Cast_1Cast;sequential_63/random_zoom_32/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ª
<sequential_63/random_zoom_32/stateful_uniform/RngReadAndSkipRngReadAndSkipEsequential_63_random_zoom_32_stateful_uniform_rngreadandskip_resource=sequential_63/random_zoom_32/stateful_uniform/Cast/x:output:08sequential_63/random_zoom_32/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Asequential_63/random_zoom_32/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Csequential_63/random_zoom_32/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Csequential_63/random_zoom_32/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
;sequential_63/random_zoom_32/stateful_uniform/strided_sliceStridedSliceDsequential_63/random_zoom_32/stateful_uniform/RngReadAndSkip:value:0Jsequential_63/random_zoom_32/stateful_uniform/strided_slice/stack:output:0Lsequential_63/random_zoom_32/stateful_uniform/strided_slice/stack_1:output:0Lsequential_63/random_zoom_32/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask·
5sequential_63/random_zoom_32/stateful_uniform/BitcastBitcastDsequential_63/random_zoom_32/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Csequential_63/random_zoom_32/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Esequential_63/random_zoom_32/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_63/random_zoom_32/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
=sequential_63/random_zoom_32/stateful_uniform/strided_slice_1StridedSliceDsequential_63/random_zoom_32/stateful_uniform/RngReadAndSkip:value:0Lsequential_63/random_zoom_32/stateful_uniform/strided_slice_1/stack:output:0Nsequential_63/random_zoom_32/stateful_uniform/strided_slice_1/stack_1:output:0Nsequential_63/random_zoom_32/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:»
7sequential_63/random_zoom_32/stateful_uniform/Bitcast_1BitcastFsequential_63/random_zoom_32/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Jsequential_63/random_zoom_32/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B : 
Fsequential_63/random_zoom_32/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_63/random_zoom_32/stateful_uniform/shape:output:0@sequential_63/random_zoom_32/stateful_uniform/Bitcast_1:output:0>sequential_63/random_zoom_32/stateful_uniform/Bitcast:output:0Ssequential_63/random_zoom_32/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
1sequential_63/random_zoom_32/stateful_uniform/subSub:sequential_63/random_zoom_32/stateful_uniform/max:output:0:sequential_63/random_zoom_32/stateful_uniform/min:output:0*
T0*
_output_shapes
: ò
1sequential_63/random_zoom_32/stateful_uniform/mulMulOsequential_63/random_zoom_32/stateful_uniform/StatelessRandomUniformV2:output:05sequential_63/random_zoom_32/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
-sequential_63/random_zoom_32/stateful_uniformAddV25sequential_63/random_zoom_32/stateful_uniform/mul:z:0:sequential_63/random_zoom_32/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(sequential_63/random_zoom_32/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
#sequential_63/random_zoom_32/concatConcatV21sequential_63/random_zoom_32/stateful_uniform:z:01sequential_63/random_zoom_32/stateful_uniform:z:01sequential_63/random_zoom_32/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.sequential_63/random_zoom_32/zoom_matrix/ShapeShape,sequential_63/random_zoom_32/concat:output:0*
T0*
_output_shapes
:
<sequential_63/random_zoom_32/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential_63/random_zoom_32/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>sequential_63/random_zoom_32/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_63/random_zoom_32/zoom_matrix/strided_sliceStridedSlice7sequential_63/random_zoom_32/zoom_matrix/Shape:output:0Esequential_63/random_zoom_32/zoom_matrix/strided_slice/stack:output:0Gsequential_63/random_zoom_32/zoom_matrix/strided_slice/stack_1:output:0Gsequential_63/random_zoom_32/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
.sequential_63/random_zoom_32/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
,sequential_63/random_zoom_32/zoom_matrix/subSub'sequential_63/random_zoom_32/Cast_1:y:07sequential_63/random_zoom_32/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: w
2sequential_63/random_zoom_32/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ë
0sequential_63/random_zoom_32/zoom_matrix/truedivRealDiv0sequential_63/random_zoom_32/zoom_matrix/sub:z:0;sequential_63/random_zoom_32/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
>sequential_63/random_zoom_32/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ã
8sequential_63/random_zoom_32/zoom_matrix/strided_slice_1StridedSlice,sequential_63/random_zoom_32/concat:output:0Gsequential_63/random_zoom_32/zoom_matrix/strided_slice_1/stack:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_1/stack_1:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masku
0sequential_63/random_zoom_32/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?å
.sequential_63/random_zoom_32/zoom_matrix/sub_1Sub9sequential_63/random_zoom_32/zoom_matrix/sub_1/x:output:0Asequential_63/random_zoom_32/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
,sequential_63/random_zoom_32/zoom_matrix/mulMul4sequential_63/random_zoom_32/zoom_matrix/truediv:z:02sequential_63/random_zoom_32/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
0sequential_63/random_zoom_32/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¸
.sequential_63/random_zoom_32/zoom_matrix/sub_2Sub%sequential_63/random_zoom_32/Cast:y:09sequential_63/random_zoom_32/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: y
4sequential_63/random_zoom_32/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ñ
2sequential_63/random_zoom_32/zoom_matrix/truediv_1RealDiv2sequential_63/random_zoom_32/zoom_matrix/sub_2:z:0=sequential_63/random_zoom_32/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
>sequential_63/random_zoom_32/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ã
8sequential_63/random_zoom_32/zoom_matrix/strided_slice_2StridedSlice,sequential_63/random_zoom_32/concat:output:0Gsequential_63/random_zoom_32/zoom_matrix/strided_slice_2/stack:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_2/stack_1:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masku
0sequential_63/random_zoom_32/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?å
.sequential_63/random_zoom_32/zoom_matrix/sub_3Sub9sequential_63/random_zoom_32/zoom_matrix/sub_3/x:output:0Asequential_63/random_zoom_32/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
.sequential_63/random_zoom_32/zoom_matrix/mul_1Mul6sequential_63/random_zoom_32/zoom_matrix/truediv_1:z:02sequential_63/random_zoom_32/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_63/random_zoom_32/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ã
8sequential_63/random_zoom_32/zoom_matrix/strided_slice_3StridedSlice,sequential_63/random_zoom_32/concat:output:0Gsequential_63/random_zoom_32/zoom_matrix/strided_slice_3/stack:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_3/stack_1:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_masky
7sequential_63/random_zoom_32/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :î
5sequential_63/random_zoom_32/zoom_matrix/zeros/packedPack?sequential_63/random_zoom_32/zoom_matrix/strided_slice:output:0@sequential_63/random_zoom_32/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:y
4sequential_63/random_zoom_32/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ç
.sequential_63/random_zoom_32/zoom_matrix/zerosFill>sequential_63/random_zoom_32/zoom_matrix/zeros/packed:output:0=sequential_63/random_zoom_32/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9sequential_63/random_zoom_32/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ò
7sequential_63/random_zoom_32/zoom_matrix/zeros_1/packedPack?sequential_63/random_zoom_32/zoom_matrix/strided_slice:output:0Bsequential_63/random_zoom_32/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:{
6sequential_63/random_zoom_32/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    í
0sequential_63/random_zoom_32/zoom_matrix/zeros_1Fill@sequential_63/random_zoom_32/zoom_matrix/zeros_1/packed:output:0?sequential_63/random_zoom_32/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>sequential_63/random_zoom_32/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
@sequential_63/random_zoom_32/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ã
8sequential_63/random_zoom_32/zoom_matrix/strided_slice_4StridedSlice,sequential_63/random_zoom_32/concat:output:0Gsequential_63/random_zoom_32/zoom_matrix/strided_slice_4/stack:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_4/stack_1:output:0Isequential_63/random_zoom_32/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask{
9sequential_63/random_zoom_32/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ò
7sequential_63/random_zoom_32/zoom_matrix/zeros_2/packedPack?sequential_63/random_zoom_32/zoom_matrix/strided_slice:output:0Bsequential_63/random_zoom_32/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:{
6sequential_63/random_zoom_32/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    í
0sequential_63/random_zoom_32/zoom_matrix/zeros_2Fill@sequential_63/random_zoom_32/zoom_matrix/zeros_2/packed:output:0?sequential_63/random_zoom_32/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4sequential_63/random_zoom_32/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ð
/sequential_63/random_zoom_32/zoom_matrix/concatConcatV2Asequential_63/random_zoom_32/zoom_matrix/strided_slice_3:output:07sequential_63/random_zoom_32/zoom_matrix/zeros:output:00sequential_63/random_zoom_32/zoom_matrix/mul:z:09sequential_63/random_zoom_32/zoom_matrix/zeros_1:output:0Asequential_63/random_zoom_32/zoom_matrix/strided_slice_4:output:02sequential_63/random_zoom_32/zoom_matrix/mul_1:z:09sequential_63/random_zoom_32/zoom_matrix/zeros_2:output:0=sequential_63/random_zoom_32/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
,sequential_63/random_zoom_32/transform/ShapeShapeZsequential_63/random_rotation_29/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:
:sequential_63/random_zoom_32/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
<sequential_63/random_zoom_32/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<sequential_63/random_zoom_32/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4sequential_63/random_zoom_32/transform/strided_sliceStridedSlice5sequential_63/random_zoom_32/transform/Shape:output:0Csequential_63/random_zoom_32/transform/strided_slice/stack:output:0Esequential_63/random_zoom_32/transform/strided_slice/stack_1:output:0Esequential_63/random_zoom_32/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:v
1sequential_63/random_zoom_32/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    å
Asequential_63/random_zoom_32/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Zsequential_63/random_rotation_29/transform/ImageProjectiveTransformV3:transformed_images:08sequential_63/random_zoom_32/zoom_matrix/concat:output:0=sequential_63/random_zoom_32/transform/strided_slice:output:0:sequential_63/random_zoom_32/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
conv2d_62/Conv2DConv2DVsequential_63/random_zoom_32/transform/ImageProjectiveTransformV3:transformed_images:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
paddingVALID*
strides

 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
n
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
^
dropout_116/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_116/dropout/MulMulconv2d_62/Relu:activations:0"dropout_116/dropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
e
dropout_116/dropout/ShapeShapeconv2d_62/Relu:activations:0*
T0*
_output_shapes
:®
0dropout_116/dropout/random_uniform/RandomUniformRandomUniform"dropout_116/dropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
dtype0g
"dropout_116/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ô
 dropout_116/dropout/GreaterEqualGreaterEqual9dropout_116/dropout/random_uniform/RandomUniform:output:0+dropout_116/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

dropout_116/dropout/CastCast$dropout_116/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

dropout_116/dropout/Mul_1Muldropout_116/dropout/Mul:z:0dropout_116/dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
¯
max_pooling2d_62/MaxPoolMaxPooldropout_116/dropout/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
^
dropout_117/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_117/dropout/MulMul!max_pooling2d_62/MaxPool:output:0"dropout_117/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
dropout_117/dropout/ShapeShape!max_pooling2d_62/MaxPool:output:0*
T0*
_output_shapes
:¬
0dropout_117/dropout/random_uniform/RandomUniformRandomUniform"dropout_117/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0g
"dropout_117/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ò
 dropout_117/dropout/GreaterEqualGreaterEqual9dropout_117/dropout/random_uniform/RandomUniform:output:0+dropout_117/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dropout_117/dropout/CastCast$dropout_117/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dropout_117/dropout/Mul_1Muldropout_117/dropout/Mul:z:0dropout_117/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0Å
conv2d_63/Conv2DConv2Ddropout_117/dropout/Mul_1:z:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
paddingVALID*
strides

 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
l
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
^
dropout_118/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_118/dropout/MulMulconv2d_63/Relu:activations:0"dropout_118/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
e
dropout_118/dropout/ShapeShapeconv2d_63/Relu:activations:0*
T0*
_output_shapes
:¬
0dropout_118/dropout/random_uniform/RandomUniformRandomUniform"dropout_118/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
dtype0g
"dropout_118/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ò
 dropout_118/dropout/GreaterEqualGreaterEqual9dropout_118/dropout/random_uniform/RandomUniform:output:0+dropout_118/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

dropout_118/dropout/CastCast$dropout_118/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

dropout_118/dropout/Mul_1Muldropout_118/dropout/Mul:z:0dropout_118/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
¯
max_pooling2d_63/MaxPoolMaxPooldropout_118/dropout/Mul_1:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
*
ksize
*
paddingVALID*
strides
^
dropout_119/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?
dropout_119/dropout/MulMul!max_pooling2d_63/MaxPool:output:0"dropout_119/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
j
dropout_119/dropout/ShapeShape!max_pooling2d_63/MaxPool:output:0*
T0*
_output_shapes
:¬
0dropout_119/dropout/random_uniform/RandomUniformRandomUniform"dropout_119/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
*
dtype0g
"dropout_119/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>Ò
 dropout_119/dropout/GreaterEqualGreaterEqual9dropout_119/dropout/random_uniform/RandomUniform:output:0+dropout_119/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

dropout_119/dropout/CastCast$dropout_119/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

dropout_119/dropout/Mul_1Muldropout_119/dropout/Mul:z:0dropout_119/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
a
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  
flatten_28/ReshapeReshapedropout_119/dropout/Mul_1:z:0flatten_28/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
¨¬*
dtype0
dense_31/MatMulMatMulflatten_28/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitydense_31/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOpF^sequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkipH^sequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkipA^sequential_63/random_rotation_29/stateful_uniform/RngReadAndSkip=^sequential_63/random_zoom_32/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2
Esequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkipEsequential_63/random_flip_32/stateful_uniform_full_int/RngReadAndSkip2
Gsequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkipGsequential_63/random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip2
@sequential_63/random_rotation_29/stateful_uniform/RngReadAndSkip@sequential_63/random_rotation_29/stateful_uniform/RngReadAndSkip2|
<sequential_63/random_zoom_32/stateful_uniform/RngReadAndSkip<sequential_63/random_zoom_32/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÎC

__inference__traced_save_77386
file_prefix/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopR
Nsavev2_sequential_64_sequential_63_random_flip_32_statevar_read_readvariableop	V
Rsavev2_sequential_64_sequential_63_random_rotation_29_statevar_read_readvariableop	R
Nsavev2_sequential_64_sequential_63_random_zoom_32_statevar_read_readvariableop	6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ñ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH«
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopNsavev2_sequential_64_sequential_63_random_flip_32_statevar_read_readvariableopRsavev2_sequential_64_sequential_63_random_rotation_29_statevar_read_readvariableopNsavev2_sequential_64_sequential_63_random_zoom_32_statevar_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2				
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesñ
î: :
:
:

:
:
¨¬:: : : : : : : : : ::::
:
:

:
:
¨¬::
:
:

:
:
¨¬:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:&"
 
_output_shapes
:
¨¬: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:&"
 
_output_shapes
:
¨¬: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:&"
 
_output_shapes
:
¨¬: 

_output_shapes
::

_output_shapes
: 
¡

ö
C__inference_dense_31_layer_call_and_return_conditional_losses_76879

inputs2
matmul_readvariableop_resource:
¨¬-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¨¬*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬
 
_user_specified_nameinputs

d
F__inference_dropout_116_layer_call_and_return_conditional_losses_75492

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ
:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

 
_user_specified_nameinputs
õ2
Þ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75859
sequential_63_input!
sequential_63_75829:	!
sequential_63_75831:	!
sequential_63_75833:	)
conv2d_62_75836:

conv2d_62_75838:
)
conv2d_63_75844:


conv2d_63_75846:
"
dense_31_75853:
¨¬
dense_31_75855:
identity¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall¢#dropout_116/StatefulPartitionedCall¢#dropout_117/StatefulPartitionedCall¢#dropout_118/StatefulPartitionedCall¢#dropout_119/StatefulPartitionedCall¢%sequential_63/StatefulPartitionedCall­
%sequential_63/StatefulPartitionedCallStatefulPartitionedCallsequential_63_inputsequential_63_75829sequential_63_75831sequential_63_75833*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75398¦
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall.sequential_63/StatefulPartitionedCall:output:0conv2d_62_75836conv2d_62_75838*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_75481þ
#dropout_116/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_116_layer_call_and_return_conditional_losses_75690ø
 max_pooling2d_62/PartitionedCallPartitionedCall,dropout_116/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_75447¡
#dropout_117/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_62/PartitionedCall:output:0$^dropout_116/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_117_layer_call_and_return_conditional_losses_75667¢
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall,dropout_117/StatefulPartitionedCall:output:0conv2d_63_75844conv2d_63_75846*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_75513¢
#dropout_118/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0$^dropout_117/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_118_layer_call_and_return_conditional_losses_75634ø
 max_pooling2d_63/PartitionedCallPartitionedCall,dropout_118/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_75459¡
#dropout_119/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_63/PartitionedCall:output:0$^dropout_118/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_119_layer_call_and_return_conditional_losses_75611æ
flatten_28/PartitionedCallPartitionedCall,dropout_119/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_75540
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_31_75853dense_31_75855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_75553x
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall$^dropout_116/StatefulPartitionedCall$^dropout_117/StatefulPartitionedCall$^dropout_118/StatefulPartitionedCall$^dropout_119/StatefulPartitionedCall&^sequential_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2J
#dropout_116/StatefulPartitionedCall#dropout_116/StatefulPartitionedCall2J
#dropout_117/StatefulPartitionedCall#dropout_117/StatefulPartitionedCall2J
#dropout_118/StatefulPartitionedCall#dropout_118/StatefulPartitionedCall2J
#dropout_119/StatefulPartitionedCall#dropout_119/StatefulPartitionedCall2N
%sequential_63/StatefulPartitionedCall%sequential_63/StatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesequential_63_input
Ä
G
+__inference_dropout_119_layer_call_fn_76826

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_119_layer_call_and_return_conditional_losses_75532h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs

d
F__inference_dropout_116_layer_call_and_return_conditional_losses_76715

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ
:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

 
_user_specified_nameinputs
É
a
E__inference_flatten_28_layer_call_and_return_conditional_losses_76859

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs
¨
r
H__inference_sequential_63_layer_call_and_return_conditional_losses_75425
random_flip_32_input
identityÞ
random_flip_32/PartitionedCallPartitionedCallrandom_flip_32_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_flip_32_layer_call_and_return_conditional_losses_74990ù
"random_rotation_29/PartitionedCallPartitionedCall'random_flip_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_74996õ
random_zoom_32/PartitionedCallPartitionedCall+random_rotation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75002y
IdentityIdentity'random_zoom_32/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namerandom_flip_32_input

d
+__inference_dropout_117_layer_call_fn_76747

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_117_layer_call_and_return_conditional_losses_75667w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
´

e
F__inference_dropout_119_layer_call_and_return_conditional_losses_75611

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs

Æ
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_75249

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
½
-__inference_sequential_63_layer_call_fn_75418
random_flip_32_input
unknown:	
	unknown_0:	
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_32_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75398y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namerandom_flip_32_input

Æ
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_77140

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


I__inference_random_flip_32_layer_call_and_return_conditional_losses_77006

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢*stateful_uniform_full_int_1/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ë
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:k
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: d
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:y
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0{
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0a
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	V
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R ~
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskÛ
0stateless_random_flip_up_down/control_dependencyIdentity(stateless_random_flip_left_right/add:z:0*
T0*7
_class-
+)loc:@stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#stateless_random_flip_up_down/ShapeShape9stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:{
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¨
<stateless_random_flip_up_down/stateless_random_uniform/shapePack4stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0* 
_output_shapes
::
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ï
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :¿
+stateless_random_flip_up_down/Reshape/shapePack4stateless_random_flip_up_down/strided_slice:output:06stateless_random_flip_up_down/Reshape/shape/1:output:06stateless_random_flip_up_down/Reshape/shape/2:output:06stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ü
%stateless_random_flip_up_down/ReshapeReshape:stateless_random_flip_up_down/stateless_random_uniform:z:04stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#stateless_random_flip_up_down/RoundRound.stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:â
'stateless_random_flip_up_down/ReverseV2	ReverseV29stateless_random_flip_up_down/control_dependency:output:05stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
!stateless_random_flip_up_down/mulMul'stateless_random_flip_up_down/Round:y:00stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
!stateless_random_flip_up_down/subSub,stateless_random_flip_up_down/sub/x:output:0'stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
#stateless_random_flip_up_down/mul_1Mul%stateless_random_flip_up_down/sub:z:09stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!stateless_random_flip_up_down/addAddV2%stateless_random_flip_up_down/mul:z:0'stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
IdentityIdentity%stateless_random_flip_up_down/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_random_flip_32_layer_call_and_return_conditional_losses_74990

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

)__inference_conv2d_62_layer_call_fn_76689

inputs!
unknown:

	unknown_0:

identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_75481y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

e
F__inference_dropout_117_layer_call_and_return_conditional_losses_76764

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

e
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75002

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

e
F__inference_dropout_116_layer_call_and_return_conditional_losses_76727

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>°
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ
:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

 
_user_specified_nameinputs

ý
D__inference_conv2d_63_layer_call_and_return_conditional_losses_75513

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ù
d
F__inference_dropout_119_layer_call_and_return_conditional_losses_75532

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs

d
+__inference_dropout_116_layer_call_fn_76710

inputs
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_116_layer_call_and_return_conditional_losses_75690y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ
22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

 
_user_specified_nameinputs
	

#__inference_signature_wrapper_76337
sequential_63_input!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:
¨¬
	unknown_4:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallsequential_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_74979o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesequential_63_input
n
Â
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75118

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_sequential_63_layer_call_and_return_conditional_losses_76357

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
d
F__inference_dropout_117_layer_call_and_return_conditional_losses_76752

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ù
d
F__inference_dropout_118_layer_call_and_return_conditional_losses_75524

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

 
_user_specified_nameinputs
Ð

2__inference_random_rotation_29_layer_call_fn_77018

inputs
unknown:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_75249y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_63_layer_call_fn_76816

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_75459
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_63_layer_call_and_return_conditional_losses_76784

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
´

e
F__inference_dropout_118_layer_call_and_return_conditional_losses_76811

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

 
_user_specified_nameinputs
¡

ö
C__inference_dense_31_layer_call_and_return_conditional_losses_75553

inputs2
matmul_readvariableop_resource:
¨¬-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¨¬*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬
 
_user_specified_nameinputs
Øz
þ
!__inference__traced_restore_77486
file_prefix;
!assignvariableop_conv2d_62_kernel:
/
!assignvariableop_1_conv2d_62_bias:
=
#assignvariableop_2_conv2d_63_kernel:

/
!assignvariableop_3_conv2d_63_bias:
6
"assignvariableop_4_dense_31_kernel:
¨¬.
 assignvariableop_5_dense_31_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: U
Gassignvariableop_15_sequential_64_sequential_63_random_flip_32_statevar:	Y
Kassignvariableop_16_sequential_64_sequential_63_random_rotation_29_statevar:	U
Gassignvariableop_17_sequential_64_sequential_63_random_zoom_32_statevar:	E
+assignvariableop_18_adam_conv2d_62_kernel_m:
7
)assignvariableop_19_adam_conv2d_62_bias_m:
E
+assignvariableop_20_adam_conv2d_63_kernel_m:

7
)assignvariableop_21_adam_conv2d_63_bias_m:
>
*assignvariableop_22_adam_dense_31_kernel_m:
¨¬6
(assignvariableop_23_adam_dense_31_bias_m:E
+assignvariableop_24_adam_conv2d_62_kernel_v:
7
)assignvariableop_25_adam_conv2d_62_bias_v:
E
+assignvariableop_26_adam_conv2d_63_kernel_v:

7
)assignvariableop_27_adam_conv2d_63_bias_v:
>
*assignvariableop_28_adam_dense_31_kernel_v:
¨¬6
(assignvariableop_29_adam_dense_31_bias_v:
identity_31¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ô
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH®
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B º
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_62_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_62_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_63_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_63_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_31_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_31_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:¸
AssignVariableOp_15AssignVariableOpGassignvariableop_15_sequential_64_sequential_63_random_flip_32_statevarIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:¼
AssignVariableOp_16AssignVariableOpKassignvariableop_16_sequential_64_sequential_63_random_rotation_29_statevarIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:¸
AssignVariableOp_17AssignVariableOpGassignvariableop_17_sequential_64_sequential_63_random_zoom_32_statevarIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_conv2d_62_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_conv2d_62_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_conv2d_63_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_conv2d_63_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_31_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_31_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_conv2d_62_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_conv2d_62_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_conv2d_63_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_conv2d_63_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_31_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_31_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ã
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: Ð
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ò
J
.__inference_random_flip_32_layer_call_fn_76884

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_flip_32_layer_call_and_return_conditional_losses_74990j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
d
F__inference_dropout_118_layer_call_and_return_conditional_losses_76799

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

 
_user_specified_nameinputs
Ä
G
+__inference_dropout_118_layer_call_fn_76789

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_118_layer_call_and_return_conditional_losses_75524h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

 
_user_specified_nameinputs
Ö
Ö
H__inference_sequential_63_layer_call_and_return_conditional_losses_75398

inputs"
random_flip_32_75388:	&
random_rotation_29_75391:	"
random_zoom_32_75394:	
identity¢&random_flip_32/StatefulPartitionedCall¢*random_rotation_29/StatefulPartitionedCall¢&random_zoom_32/StatefulPartitionedCall÷
&random_flip_32/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_flip_32_75388*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_flip_32_layer_call_and_return_conditional_losses_75373¬
*random_rotation_29/StatefulPartitionedCallStatefulPartitionedCall/random_flip_32/StatefulPartitionedCall:output:0random_rotation_29_75391*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_75249¤
&random_zoom_32/StatefulPartitionedCallStatefulPartitionedCall3random_rotation_29/StatefulPartitionedCall:output:0random_zoom_32_75394*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75118
IdentityIdentity/random_zoom_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
NoOpNoOp'^random_flip_32/StatefulPartitionedCall+^random_rotation_29/StatefulPartitionedCall'^random_zoom_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : 2P
&random_flip_32/StatefulPartitionedCall&random_flip_32/StatefulPartitionedCall2X
*random_rotation_29/StatefulPartitionedCall*random_rotation_29/StatefulPartitionedCall2P
&random_zoom_32/StatefulPartitionedCall&random_zoom_32/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
W
-__inference_sequential_63_layer_call_fn_75008
random_flip_32_input
identityÎ
PartitionedCallPartitionedCallrandom_flip_32_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75005j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namerandom_flip_32_input
Â	
¥
-__inference_sequential_64_layer_call_fn_75575
sequential_63_input!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:
¨¬
	unknown_4:
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallsequential_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesequential_63_input


I__inference_random_flip_32_layer_call_and_return_conditional_losses_75373

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢*stateful_uniform_full_int_1/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ë
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:k
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¡
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: d
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:y
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ã
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0{
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0a
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	V
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R ~
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskÛ
0stateless_random_flip_up_down/control_dependencyIdentity(stateless_random_flip_left_right/add:z:0*
T0*7
_class-
+)loc:@stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#stateless_random_flip_up_down/ShapeShape9stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:{
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¨
<stateless_random_flip_up_down/stateless_random_uniform/shapePack4stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¯
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0* 
_output_shapes
::
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ï
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :o
-stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :¿
+stateless_random_flip_up_down/Reshape/shapePack4stateless_random_flip_up_down/strided_slice:output:06stateless_random_flip_up_down/Reshape/shape/1:output:06stateless_random_flip_up_down/Reshape/shape/2:output:06stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Ü
%stateless_random_flip_up_down/ReshapeReshape:stateless_random_flip_up_down/stateless_random_uniform:z:04stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#stateless_random_flip_up_down/RoundRound.stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:â
'stateless_random_flip_up_down/ReverseV2	ReverseV29stateless_random_flip_up_down/control_dependency:output:05stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
!stateless_random_flip_up_down/mulMul'stateless_random_flip_up_down/Round:y:00stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
#stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
!stateless_random_flip_up_down/subSub,stateless_random_flip_up_down/sub/x:output:0'stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
#stateless_random_flip_up_down/mul_1Mul%stateless_random_flip_up_down/sub:z:09stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!stateless_random_flip_up_down/addAddV2%stateless_random_flip_up_down/mul:z:0'stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
IdentityIdentity%stateless_random_flip_up_down/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
~
.__inference_random_zoom_32_layer_call_fn_77152

inputs
unknown:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75118y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_62_layer_call_and_return_conditional_losses_76700

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
~
.__inference_random_flip_32_layer_call_fn_76891

inputs
unknown:	
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_flip_32_layer_call_and_return_conditional_losses_75373y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

e
F__inference_dropout_117_layer_call_and_return_conditional_losses_75667

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

e
I__inference_random_flip_32_layer_call_and_return_conditional_losses_76895

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú)
µ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75826
sequential_63_input)
conv2d_62_75803:

conv2d_62_75805:
)
conv2d_63_75811:


conv2d_63_75813:
"
dense_31_75820:
¨¬
dense_31_75822:
identity¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCallÛ
sequential_63/PartitionedCallPartitionedCallsequential_63_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75005
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&sequential_63/PartitionedCall:output:0conv2d_62_75803conv2d_62_75805*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_75481î
dropout_116/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_116_layer_call_and_return_conditional_losses_75492ð
 max_pooling2d_62/PartitionedCallPartitionedCall$dropout_116/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_75447ë
dropout_117/PartitionedCallPartitionedCall)max_pooling2d_62/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_117_layer_call_and_return_conditional_losses_75500
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall$dropout_117/PartitionedCall:output:0conv2d_63_75811conv2d_63_75813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_75513ì
dropout_118/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_118_layer_call_and_return_conditional_losses_75524ð
 max_pooling2d_63/PartitionedCallPartitionedCall$dropout_118/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_75459ë
dropout_119/PartitionedCallPartitionedCall)max_pooling2d_63/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_119_layer_call_and_return_conditional_losses_75532Þ
flatten_28/PartitionedCallPartitionedCall$dropout_119/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_75540
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_31_75820dense_31_75822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_75553x
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesequential_63_input
Ú/
°
 __inference__wrapped_model_74979
sequential_63_inputP
6sequential_64_conv2d_62_conv2d_readvariableop_resource:
E
7sequential_64_conv2d_62_biasadd_readvariableop_resource:
P
6sequential_64_conv2d_63_conv2d_readvariableop_resource:

E
7sequential_64_conv2d_63_biasadd_readvariableop_resource:
I
5sequential_64_dense_31_matmul_readvariableop_resource:
¨¬D
6sequential_64_dense_31_biasadd_readvariableop_resource:
identity¢.sequential_64/conv2d_62/BiasAdd/ReadVariableOp¢-sequential_64/conv2d_62/Conv2D/ReadVariableOp¢.sequential_64/conv2d_63/BiasAdd/ReadVariableOp¢-sequential_64/conv2d_63/Conv2D/ReadVariableOp¢-sequential_64/dense_31/BiasAdd/ReadVariableOp¢,sequential_64/dense_31/MatMul/ReadVariableOp¬
-sequential_64/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_64_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Ù
sequential_64/conv2d_62/Conv2DConv2Dsequential_63_input5sequential_64/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
paddingVALID*
strides
¢
.sequential_64/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ç
sequential_64/conv2d_62/BiasAddBiasAdd'sequential_64/conv2d_62/Conv2D:output:06sequential_64/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

sequential_64/conv2d_62/ReluRelu(sequential_64/conv2d_62/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

"sequential_64/dropout_116/IdentityIdentity*sequential_64/conv2d_62/Relu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
Ë
&sequential_64/max_pooling2d_62/MaxPoolMaxPool+sequential_64/dropout_116/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides

"sequential_64/dropout_117/IdentityIdentity/sequential_64/max_pooling2d_62/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
-sequential_64/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_64_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0ï
sequential_64/conv2d_63/Conv2DConv2D+sequential_64/dropout_117/Identity:output:05sequential_64/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
paddingVALID*
strides
¢
.sequential_64/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_64_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Å
sequential_64/conv2d_63/BiasAddBiasAdd'sequential_64/conv2d_63/Conv2D:output:06sequential_64/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

sequential_64/conv2d_63/ReluRelu(sequential_64/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

"sequential_64/dropout_118/IdentityIdentity*sequential_64/conv2d_63/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
Ë
&sequential_64/max_pooling2d_63/MaxPoolMaxPool+sequential_64/dropout_118/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
*
ksize
*
paddingVALID*
strides

"sequential_64/dropout_119/IdentityIdentity/sequential_64/max_pooling2d_63/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
o
sequential_64/flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  µ
 sequential_64/flatten_28/ReshapeReshape+sequential_64/dropout_119/Identity:output:0'sequential_64/flatten_28/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬¤
,sequential_64/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_64_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
¨¬*
dtype0º
sequential_64/dense_31/MatMulMatMul)sequential_64/flatten_28/Reshape:output:04sequential_64/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_64/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_64_dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_64/dense_31/BiasAddBiasAdd'sequential_64/dense_31/MatMul:product:05sequential_64/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_64/dense_31/SigmoidSigmoid'sequential_64/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
IdentityIdentity"sequential_64/dense_31/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
NoOpNoOp/^sequential_64/conv2d_62/BiasAdd/ReadVariableOp.^sequential_64/conv2d_62/Conv2D/ReadVariableOp/^sequential_64/conv2d_63/BiasAdd/ReadVariableOp.^sequential_64/conv2d_63/Conv2D/ReadVariableOp.^sequential_64/dense_31/BiasAdd/ReadVariableOp-^sequential_64/dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 2`
.sequential_64/conv2d_62/BiasAdd/ReadVariableOp.sequential_64/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_64/conv2d_62/Conv2D/ReadVariableOp-sequential_64/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_64/conv2d_63/BiasAdd/ReadVariableOp.sequential_64/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_64/conv2d_63/Conv2D/ReadVariableOp-sequential_64/conv2d_63/Conv2D/ReadVariableOp2^
-sequential_64/dense_31/BiasAdd/ReadVariableOp-sequential_64/dense_31/BiasAdd/ReadVariableOp2\
,sequential_64/dense_31/MatMul/ReadVariableOp,sequential_64/dense_31/MatMul/ReadVariableOp:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesequential_63_input
Ä
G
+__inference_dropout_117_layer_call_fn_76742

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_117_layer_call_and_return_conditional_losses_75500h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
þ
d
H__inference_sequential_63_layer_call_and_return_conditional_losses_75005

inputs
identityÐ
random_flip_32/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_flip_32_layer_call_and_return_conditional_losses_74990ù
"random_rotation_29/PartitionedCallPartitionedCall'random_flip_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_74996õ
random_zoom_32/PartitionedCallPartitionedCall+random_rotation_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75002y
IdentityIdentity'random_zoom_32/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ä
H__inference_sequential_63_layer_call_and_return_conditional_losses_75438
random_flip_32_input"
random_flip_32_75428:	&
random_rotation_29_75431:	"
random_zoom_32_75434:	
identity¢&random_flip_32/StatefulPartitionedCall¢*random_rotation_29/StatefulPartitionedCall¢&random_zoom_32/StatefulPartitionedCall
&random_flip_32/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_32_inputrandom_flip_32_75428*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_flip_32_layer_call_and_return_conditional_losses_75373¬
*random_rotation_29/StatefulPartitionedCallStatefulPartitionedCall/random_flip_32/StatefulPartitionedCall:output:0random_rotation_29_75431*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_75249¤
&random_zoom_32/StatefulPartitionedCallStatefulPartitionedCall3random_rotation_29/StatefulPartitionedCall:output:0random_zoom_32_75434*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75118
IdentityIdentity/random_zoom_32/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
NoOpNoOp'^random_flip_32/StatefulPartitionedCall+^random_rotation_29/StatefulPartitionedCall'^random_zoom_32/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : 2P
&random_flip_32/StatefulPartitionedCall&random_flip_32/StatefulPartitionedCall2X
*random_rotation_29/StatefulPartitionedCall*random_rotation_29/StatefulPartitionedCall2P
&random_zoom_32/StatefulPartitionedCall&random_zoom_32/StatefulPartitionedCall:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_namerandom_flip_32_input

i
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_74996

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
G
+__inference_dropout_116_layer_call_fn_76705

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_116_layer_call_and_return_conditional_losses_75492j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ
:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

 
_user_specified_nameinputs
Ò
J
.__inference_random_zoom_32_layer_call_fn_77145

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_75002j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
a
E__inference_flatten_28_layer_call_and_return_conditional_losses_75540

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs

g
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_76821

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

e
F__inference_dropout_119_layer_call_and_return_conditional_losses_76848

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>

 
_user_specified_nameinputs

e
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_77156

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¯
-__inference_sequential_63_layer_call_fn_76353

inputs
unknown:	
	unknown_0:	
	unknown_1:	
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75398y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

e
F__inference_dropout_116_layer_call_and_return_conditional_losses_75690

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>°
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ
:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ

 
_user_specified_nameinputs
·

ð
-__inference_sequential_64_layer_call_fn_75799
sequential_63_input
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:
¨¬
	unknown_7:
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallsequential_63_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75755o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-
_user_specified_namesequential_63_input
³)
¨
H__inference_sequential_64_layer_call_and_return_conditional_losses_75560

inputs)
conv2d_62_75482:

conv2d_62_75484:
)
conv2d_63_75514:


conv2d_63_75516:
"
dense_31_75554:
¨¬
dense_31_75556:
identity¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCallÎ
sequential_63/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75005
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall&sequential_63/PartitionedCall:output:0conv2d_62_75482conv2d_62_75484*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_75481î
dropout_116/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_116_layer_call_and_return_conditional_losses_75492ð
 max_pooling2d_62/PartitionedCallPartitionedCall$dropout_116/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_75447ë
dropout_117/PartitionedCallPartitionedCall)max_pooling2d_62/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_117_layer_call_and_return_conditional_losses_75500
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall$dropout_117/PartitionedCall:output:0conv2d_63_75514conv2d_63_75516*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_75513ì
dropout_118/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_118_layer_call_and_return_conditional_losses_75524ð
 max_pooling2d_63/PartitionedCallPartitionedCall$dropout_118/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_75459ë
dropout_119/PartitionedCallPartitionedCall)max_pooling2d_63/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_119_layer_call_and_return_conditional_losses_75532Þ
flatten_28/PartitionedCallPartitionedCall$dropout_119/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_75540
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_31_75554dense_31_75556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_75553x
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_62_layer_call_fn_76732

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_75447
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
d
F__inference_dropout_117_layer_call_and_return_conditional_losses_75500

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

ý
D__inference_conv2d_62_layer_call_and_return_conditional_losses_75481

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_75447

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

e
F__inference_dropout_118_layer_call_and_return_conditional_losses_75634

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}

 
_user_specified_nameinputs
Î2
Ñ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75755

inputs!
sequential_63_75725:	!
sequential_63_75727:	!
sequential_63_75729:	)
conv2d_62_75732:

conv2d_62_75734:
)
conv2d_63_75740:


conv2d_63_75742:
"
dense_31_75749:
¨¬
dense_31_75751:
identity¢!conv2d_62/StatefulPartitionedCall¢!conv2d_63/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall¢#dropout_116/StatefulPartitionedCall¢#dropout_117/StatefulPartitionedCall¢#dropout_118/StatefulPartitionedCall¢#dropout_119/StatefulPartitionedCall¢%sequential_63/StatefulPartitionedCall 
%sequential_63/StatefulPartitionedCallStatefulPartitionedCallinputssequential_63_75725sequential_63_75727sequential_63_75729*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75398¦
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall.sequential_63/StatefulPartitionedCall:output:0conv2d_62_75732conv2d_62_75734*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_75481þ
#dropout_116/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_116_layer_call_and_return_conditional_losses_75690ø
 max_pooling2d_62/PartitionedCallPartitionedCall,dropout_116/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_75447¡
#dropout_117/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_62/PartitionedCall:output:0$^dropout_116/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_117_layer_call_and_return_conditional_losses_75667¢
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall,dropout_117/StatefulPartitionedCall:output:0conv2d_63_75740conv2d_63_75742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_75513¢
#dropout_118/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0$^dropout_117/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_118_layer_call_and_return_conditional_losses_75634ø
 max_pooling2d_63/PartitionedCallPartitionedCall,dropout_118/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_75459¡
#dropout_119/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_63/PartitionedCall:output:0$^dropout_118/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dropout_119_layer_call_and_return_conditional_losses_75611æ
flatten_28/PartitionedCallPartitionedCall,dropout_119/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_28_layer_call_and_return_conditional_losses_75540
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_28/PartitionedCall:output:0dense_31_75749dense_31_75751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_75553x
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
NoOpNoOp"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall$^dropout_116/StatefulPartitionedCall$^dropout_117/StatefulPartitionedCall$^dropout_118/StatefulPartitionedCall$^dropout_119/StatefulPartitionedCall&^sequential_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2J
#dropout_116/StatefulPartitionedCall#dropout_116/StatefulPartitionedCall2J
#dropout_117/StatefulPartitionedCall#dropout_117/StatefulPartitionedCall2J
#dropout_118/StatefulPartitionedCall#dropout_118/StatefulPartitionedCall2J
#dropout_119/StatefulPartitionedCall#dropout_119/StatefulPartitionedCall2N
%sequential_63/StatefulPartitionedCall%sequential_63/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ã
-__inference_sequential_64_layer_call_fn_75905

inputs
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:

	unknown_3:
#
	unknown_4:


	unknown_5:

	unknown_6:
¨¬
	unknown_7:
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75755o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
N
2__inference_random_rotation_29_layer_call_fn_77011

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_74996j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

-__inference_sequential_64_layer_call_fn_75882

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:
¨¬
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_64_layer_call_and_return_conditional_losses_75560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_77022

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_75459

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
èÙ
¥
H__inference_sequential_63_layer_call_and_return_conditional_losses_76680

inputsN
@random_flip_32_stateful_uniform_full_int_rngreadandskip_resource:	I
;random_rotation_29_stateful_uniform_rngreadandskip_resource:	E
7random_zoom_32_stateful_uniform_rngreadandskip_resource:	
identity¢7random_flip_32/stateful_uniform_full_int/RngReadAndSkip¢9random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip¢2random_rotation_29/stateful_uniform/RngReadAndSkip¢.random_zoom_32/stateful_uniform/RngReadAndSkipx
.random_flip_32/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:x
.random_flip_32/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: È
-random_flip_32/stateful_uniform_full_int/ProdProd7random_flip_32/stateful_uniform_full_int/shape:output:07random_flip_32/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: q
/random_flip_32/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
/random_flip_32/stateful_uniform_full_int/Cast_1Cast6random_flip_32/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
7random_flip_32/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip@random_flip_32_stateful_uniform_full_int_rngreadandskip_resource8random_flip_32/stateful_uniform_full_int/Cast/x:output:03random_flip_32/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
<random_flip_32/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>random_flip_32/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>random_flip_32/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
6random_flip_32/stateful_uniform_full_int/strided_sliceStridedSlice?random_flip_32/stateful_uniform_full_int/RngReadAndSkip:value:0Erandom_flip_32/stateful_uniform_full_int/strided_slice/stack:output:0Grandom_flip_32/stateful_uniform_full_int/strided_slice/stack_1:output:0Grandom_flip_32/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask­
0random_flip_32/stateful_uniform_full_int/BitcastBitcast?random_flip_32/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
>random_flip_32/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
@random_flip_32/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@random_flip_32/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
8random_flip_32/stateful_uniform_full_int/strided_slice_1StridedSlice?random_flip_32/stateful_uniform_full_int/RngReadAndSkip:value:0Grandom_flip_32/stateful_uniform_full_int/strided_slice_1/stack:output:0Irandom_flip_32/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Irandom_flip_32/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:±
2random_flip_32/stateful_uniform_full_int/Bitcast_1BitcastArandom_flip_32/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0n
,random_flip_32/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :Ü
(random_flip_32/stateful_uniform_full_intStatelessRandomUniformFullIntV27random_flip_32/stateful_uniform_full_int/shape:output:0;random_flip_32/stateful_uniform_full_int/Bitcast_1:output:09random_flip_32/stateful_uniform_full_int/Bitcast:output:05random_flip_32/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	c
random_flip_32/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ¥
random_flip_32/stackPack1random_flip_32/stateful_uniform_full_int:output:0"random_flip_32/zeros_like:output:0*
N*
T0	*
_output_shapes

:s
"random_flip_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        u
$random_flip_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$random_flip_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
random_flip_32/strided_sliceStridedSlicerandom_flip_32/stack:output:0+random_flip_32/strided_slice/stack:output:0-random_flip_32/strided_slice/stack_1:output:0-random_flip_32/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask­
Brandom_flip_32/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
5random_flip_32/stateless_random_flip_left_right/ShapeShapeKrandom_flip_32/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Crandom_flip_32/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Erandom_flip_32/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Erandom_flip_32/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=random_flip_32/stateless_random_flip_left_right/strided_sliceStridedSlice>random_flip_32/stateless_random_flip_left_right/Shape:output:0Lrandom_flip_32/stateless_random_flip_left_right/strided_slice/stack:output:0Nrandom_flip_32/stateless_random_flip_left_right/strided_slice/stack_1:output:0Nrandom_flip_32/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
Nrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/shapePackFrandom_flip_32/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Lrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Lrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Î
erandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter%random_flip_32/strided_slice:output:0* 
_output_shapes
::§
erandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :É
arandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Wrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0krandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0orandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0nrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
Lrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/subSubUrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Urandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¿
Lrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/mulMuljrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Prandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
Hrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniformAddV2Prandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Urandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
?random_flip_32/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?random_flip_32/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
?random_flip_32/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
=random_flip_32/stateless_random_flip_left_right/Reshape/shapePackFrandom_flip_32/stateless_random_flip_left_right/strided_slice:output:0Hrandom_flip_32/stateless_random_flip_left_right/Reshape/shape/1:output:0Hrandom_flip_32/stateless_random_flip_left_right/Reshape/shape/2:output:0Hrandom_flip_32/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
7random_flip_32/stateless_random_flip_left_right/ReshapeReshapeLrandom_flip_32/stateless_random_flip_left_right/stateless_random_uniform:z:0Frandom_flip_32/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
5random_flip_32/stateless_random_flip_left_right/RoundRound@random_flip_32/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>random_flip_32/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
9random_flip_32/stateless_random_flip_left_right/ReverseV2	ReverseV2Krandom_flip_32/stateless_random_flip_left_right/control_dependency:output:0Grandom_flip_32/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
3random_flip_32/stateless_random_flip_left_right/mulMul9random_flip_32/stateless_random_flip_left_right/Round:y:0Brandom_flip_32/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5random_flip_32/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ï
3random_flip_32/stateless_random_flip_left_right/subSub>random_flip_32/stateless_random_flip_left_right/sub/x:output:09random_flip_32/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
5random_flip_32/stateless_random_flip_left_right/mul_1Mul7random_flip_32/stateless_random_flip_left_right/sub:z:0Krandom_flip_32/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
3random_flip_32/stateless_random_flip_left_right/addAddV27random_flip_32/stateless_random_flip_left_right/mul:z:09random_flip_32/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
0random_flip_32/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:z
0random_flip_32/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: Î
/random_flip_32/stateful_uniform_full_int_1/ProdProd9random_flip_32/stateful_uniform_full_int_1/shape:output:09random_flip_32/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: s
1random_flip_32/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :£
1random_flip_32/stateful_uniform_full_int_1/Cast_1Cast8random_flip_32/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ö
9random_flip_32/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip@random_flip_32_stateful_uniform_full_int_rngreadandskip_resource:random_flip_32/stateful_uniform_full_int_1/Cast/x:output:05random_flip_32/stateful_uniform_full_int_1/Cast_1:y:08^random_flip_32/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:
>random_flip_32/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
@random_flip_32/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@random_flip_32/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
8random_flip_32/stateful_uniform_full_int_1/strided_sliceStridedSliceArandom_flip_32/stateful_uniform_full_int_1/RngReadAndSkip:value:0Grandom_flip_32/stateful_uniform_full_int_1/strided_slice/stack:output:0Irandom_flip_32/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Irandom_flip_32/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask±
2random_flip_32/stateful_uniform_full_int_1/BitcastBitcastArandom_flip_32/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
@random_flip_32/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Brandom_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Brandom_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
:random_flip_32/stateful_uniform_full_int_1/strided_slice_1StridedSliceArandom_flip_32/stateful_uniform_full_int_1/RngReadAndSkip:value:0Irandom_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Krandom_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Krandom_flip_32/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:µ
4random_flip_32/stateful_uniform_full_int_1/Bitcast_1BitcastCrandom_flip_32/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0p
.random_flip_32/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :æ
*random_flip_32/stateful_uniform_full_int_1StatelessRandomUniformFullIntV29random_flip_32/stateful_uniform_full_int_1/shape:output:0=random_flip_32/stateful_uniform_full_int_1/Bitcast_1:output:0;random_flip_32/stateful_uniform_full_int_1/Bitcast:output:07random_flip_32/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	e
random_flip_32/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R «
random_flip_32/stack_1Pack3random_flip_32/stateful_uniform_full_int_1:output:0$random_flip_32/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:u
$random_flip_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&random_flip_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       w
&random_flip_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ì
random_flip_32/strided_slice_1StridedSlicerandom_flip_32/stack_1:output:0-random_flip_32/strided_slice_1/stack:output:0/random_flip_32/strided_slice_1/stack_1:output:0/random_flip_32/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
?random_flip_32/stateless_random_flip_up_down/control_dependencyIdentity7random_flip_32/stateless_random_flip_left_right/add:z:0*
T0*F
_class<
:8loc:@random_flip_32/stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2random_flip_32/stateless_random_flip_up_down/ShapeShapeHrandom_flip_32/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:
@random_flip_32/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Brandom_flip_32/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Brandom_flip_32/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:²
:random_flip_32/stateless_random_flip_up_down/strided_sliceStridedSlice;random_flip_32/stateless_random_flip_up_down/Shape:output:0Irandom_flip_32/stateless_random_flip_up_down/strided_slice/stack:output:0Krandom_flip_32/stateless_random_flip_up_down/strided_slice/stack_1:output:0Krandom_flip_32/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÆ
Krandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/shapePackCrandom_flip_32/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:
Irandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Irandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Í
brandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter'random_flip_32/strided_slice_1:output:0* 
_output_shapes
::¤
brandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :º
^random_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0hrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0krandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Irandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/subSubRrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Rrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¶
Irandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/mulMulgrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Erandom_flip_32/stateless_random_flip_up_down/stateless_random_uniformAddV2Mrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Rrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
<random_flip_32/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip_32/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :~
<random_flip_32/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
:random_flip_32/stateless_random_flip_up_down/Reshape/shapePackCrandom_flip_32/stateless_random_flip_up_down/strided_slice:output:0Erandom_flip_32/stateless_random_flip_up_down/Reshape/shape/1:output:0Erandom_flip_32/stateless_random_flip_up_down/Reshape/shape/2:output:0Erandom_flip_32/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
4random_flip_32/stateless_random_flip_up_down/ReshapeReshapeIrandom_flip_32/stateless_random_flip_up_down/stateless_random_uniform:z:0Crandom_flip_32/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
2random_flip_32/stateless_random_flip_up_down/RoundRound=random_flip_32/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;random_flip_32/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
6random_flip_32/stateless_random_flip_up_down/ReverseV2	ReverseV2Hrandom_flip_32/stateless_random_flip_up_down/control_dependency:output:0Drandom_flip_32/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
0random_flip_32/stateless_random_flip_up_down/mulMul6random_flip_32/stateless_random_flip_up_down/Round:y:0?random_flip_32/stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
2random_flip_32/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?æ
0random_flip_32/stateless_random_flip_up_down/subSub;random_flip_32/stateless_random_flip_up_down/sub/x:output:06random_flip_32/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
2random_flip_32/stateless_random_flip_up_down/mul_1Mul4random_flip_32/stateless_random_flip_up_down/sub:z:0Hrandom_flip_32/stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
0random_flip_32/stateless_random_flip_up_down/addAddV24random_flip_32/stateless_random_flip_up_down/mul:z:06random_flip_32/stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
random_rotation_29/ShapeShape4random_flip_32/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:p
&random_rotation_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(random_rotation_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(random_rotation_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 random_rotation_29/strided_sliceStridedSlice!random_rotation_29/Shape:output:0/random_rotation_29/strided_slice/stack:output:01random_rotation_29/strided_slice/stack_1:output:01random_rotation_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
(random_rotation_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ}
*random_rotation_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿt
*random_rotation_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"random_rotation_29/strided_slice_1StridedSlice!random_rotation_29/Shape:output:01random_rotation_29/strided_slice_1/stack:output:03random_rotation_29/strided_slice_1/stack_1:output:03random_rotation_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
random_rotation_29/CastCast+random_rotation_29/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: {
(random_rotation_29/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ}
*random_rotation_29/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*random_rotation_29/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"random_rotation_29/strided_slice_2StridedSlice!random_rotation_29/Shape:output:01random_rotation_29/strided_slice_2/stack:output:03random_rotation_29/strided_slice_2/stack_1:output:03random_rotation_29/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
random_rotation_29/Cast_1Cast+random_rotation_29/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 
)random_rotation_29/stateful_uniform/shapePack)random_rotation_29/strided_slice:output:0*
N*
T0*
_output_shapes
:l
'random_rotation_29/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿l
'random_rotation_29/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?s
)random_rotation_29/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¹
(random_rotation_29/stateful_uniform/ProdProd2random_rotation_29/stateful_uniform/shape:output:02random_rotation_29/stateful_uniform/Const:output:0*
T0*
_output_shapes
: l
*random_rotation_29/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
*random_rotation_29/stateful_uniform/Cast_1Cast1random_rotation_29/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
2random_rotation_29/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_rotation_29_stateful_uniform_rngreadandskip_resource3random_rotation_29/stateful_uniform/Cast/x:output:0.random_rotation_29/stateful_uniform/Cast_1:y:0*
_output_shapes
:
7random_rotation_29/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9random_rotation_29/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9random_rotation_29/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1random_rotation_29/stateful_uniform/strided_sliceStridedSlice:random_rotation_29/stateful_uniform/RngReadAndSkip:value:0@random_rotation_29/stateful_uniform/strided_slice/stack:output:0Brandom_rotation_29/stateful_uniform/strided_slice/stack_1:output:0Brandom_rotation_29/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask£
+random_rotation_29/stateful_uniform/BitcastBitcast:random_rotation_29/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
9random_rotation_29/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;random_rotation_29/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;random_rotation_29/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3random_rotation_29/stateful_uniform/strided_slice_1StridedSlice:random_rotation_29/stateful_uniform/RngReadAndSkip:value:0Brandom_rotation_29/stateful_uniform/strided_slice_1/stack:output:0Drandom_rotation_29/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_rotation_29/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:§
-random_rotation_29/stateful_uniform/Bitcast_1Bitcast<random_rotation_29/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
@random_rotation_29/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :ê
<random_rotation_29/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV22random_rotation_29/stateful_uniform/shape:output:06random_rotation_29/stateful_uniform/Bitcast_1:output:04random_rotation_29/stateful_uniform/Bitcast:output:0Irandom_rotation_29/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'random_rotation_29/stateful_uniform/subSub0random_rotation_29/stateful_uniform/max:output:00random_rotation_29/stateful_uniform/min:output:0*
T0*
_output_shapes
: Ð
'random_rotation_29/stateful_uniform/mulMulErandom_rotation_29/stateful_uniform/StatelessRandomUniformV2:output:0+random_rotation_29/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
#random_rotation_29/stateful_uniformAddV2+random_rotation_29/stateful_uniform/mul:z:00random_rotation_29/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(random_rotation_29/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ? 
&random_rotation_29/rotation_matrix/subSubrandom_rotation_29/Cast_1:y:01random_rotation_29/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 
&random_rotation_29/rotation_matrix/CosCos'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*random_rotation_29/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
(random_rotation_29/rotation_matrix/sub_1Subrandom_rotation_29/Cast_1:y:03random_rotation_29/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: µ
&random_rotation_29/rotation_matrix/mulMul*random_rotation_29/rotation_matrix/Cos:y:0,random_rotation_29/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&random_rotation_29/rotation_matrix/SinSin'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*random_rotation_29/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
(random_rotation_29/rotation_matrix/sub_2Subrandom_rotation_29/Cast:y:03random_rotation_29/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ·
(random_rotation_29/rotation_matrix/mul_1Mul*random_rotation_29/rotation_matrix/Sin:y:0,random_rotation_29/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
(random_rotation_29/rotation_matrix/sub_3Sub*random_rotation_29/rotation_matrix/mul:z:0,random_rotation_29/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
(random_rotation_29/rotation_matrix/sub_4Sub*random_rotation_29/rotation_matrix/sub:z:0,random_rotation_29/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
,random_rotation_29/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @È
*random_rotation_29/rotation_matrix/truedivRealDiv,random_rotation_29/rotation_matrix/sub_4:z:05random_rotation_29/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*random_rotation_29/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
(random_rotation_29/rotation_matrix/sub_5Subrandom_rotation_29/Cast:y:03random_rotation_29/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 
(random_rotation_29/rotation_matrix/Sin_1Sin'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*random_rotation_29/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
(random_rotation_29/rotation_matrix/sub_6Subrandom_rotation_29/Cast_1:y:03random_rotation_29/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ¹
(random_rotation_29/rotation_matrix/mul_2Mul,random_rotation_29/rotation_matrix/Sin_1:y:0,random_rotation_29/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(random_rotation_29/rotation_matrix/Cos_1Cos'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*random_rotation_29/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
(random_rotation_29/rotation_matrix/sub_7Subrandom_rotation_29/Cast:y:03random_rotation_29/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ¹
(random_rotation_29/rotation_matrix/mul_3Mul,random_rotation_29/rotation_matrix/Cos_1:y:0,random_rotation_29/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
&random_rotation_29/rotation_matrix/addAddV2,random_rotation_29/rotation_matrix/mul_2:z:0,random_rotation_29/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
(random_rotation_29/rotation_matrix/sub_8Sub,random_rotation_29/rotation_matrix/sub_5:z:0*random_rotation_29/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
.random_rotation_29/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ì
,random_rotation_29/rotation_matrix/truediv_1RealDiv,random_rotation_29/rotation_matrix/sub_8:z:07random_rotation_29/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(random_rotation_29/rotation_matrix/ShapeShape'random_rotation_29/stateful_uniform:z:0*
T0*
_output_shapes
:
6random_rotation_29/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8random_rotation_29/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8random_rotation_29/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0random_rotation_29/rotation_matrix/strided_sliceStridedSlice1random_rotation_29/rotation_matrix/Shape:output:0?random_rotation_29/rotation_matrix/strided_slice/stack:output:0Arandom_rotation_29/rotation_matrix/strided_slice/stack_1:output:0Arandom_rotation_29/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(random_rotation_29/rotation_matrix/Cos_2Cos'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8random_rotation_29/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
2random_rotation_29/rotation_matrix/strided_slice_1StridedSlice,random_rotation_29/rotation_matrix/Cos_2:y:0Arandom_rotation_29/rotation_matrix/strided_slice_1/stack:output:0Crandom_rotation_29/rotation_matrix/strided_slice_1/stack_1:output:0Crandom_rotation_29/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(random_rotation_29/rotation_matrix/Sin_2Sin'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8random_rotation_29/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
2random_rotation_29/rotation_matrix/strided_slice_2StridedSlice,random_rotation_29/rotation_matrix/Sin_2:y:0Arandom_rotation_29/rotation_matrix/strided_slice_2/stack:output:0Crandom_rotation_29/rotation_matrix/strided_slice_2/stack_1:output:0Crandom_rotation_29/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
&random_rotation_29/rotation_matrix/NegNeg;random_rotation_29/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8random_rotation_29/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      µ
2random_rotation_29/rotation_matrix/strided_slice_3StridedSlice.random_rotation_29/rotation_matrix/truediv:z:0Arandom_rotation_29/rotation_matrix/strided_slice_3/stack:output:0Crandom_rotation_29/rotation_matrix/strided_slice_3/stack_1:output:0Crandom_rotation_29/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(random_rotation_29/rotation_matrix/Sin_3Sin'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8random_rotation_29/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
2random_rotation_29/rotation_matrix/strided_slice_4StridedSlice,random_rotation_29/rotation_matrix/Sin_3:y:0Arandom_rotation_29/rotation_matrix/strided_slice_4/stack:output:0Crandom_rotation_29/rotation_matrix/strided_slice_4/stack_1:output:0Crandom_rotation_29/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(random_rotation_29/rotation_matrix/Cos_3Cos'random_rotation_29/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8random_rotation_29/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ³
2random_rotation_29/rotation_matrix/strided_slice_5StridedSlice,random_rotation_29/rotation_matrix/Cos_3:y:0Arandom_rotation_29/rotation_matrix/strided_slice_5/stack:output:0Crandom_rotation_29/rotation_matrix/strided_slice_5/stack_1:output:0Crandom_rotation_29/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
8random_rotation_29/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:random_rotation_29/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ·
2random_rotation_29/rotation_matrix/strided_slice_6StridedSlice0random_rotation_29/rotation_matrix/truediv_1:z:0Arandom_rotation_29/rotation_matrix/strided_slice_6/stack:output:0Crandom_rotation_29/rotation_matrix/strided_slice_6/stack_1:output:0Crandom_rotation_29/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_masks
1random_rotation_29/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ü
/random_rotation_29/rotation_matrix/zeros/packedPack9random_rotation_29/rotation_matrix/strided_slice:output:0:random_rotation_29/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:s
.random_rotation_29/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Õ
(random_rotation_29/rotation_matrix/zerosFill8random_rotation_29/rotation_matrix/zeros/packed:output:07random_rotation_29/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.random_rotation_29/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¹
)random_rotation_29/rotation_matrix/concatConcatV2;random_rotation_29/rotation_matrix/strided_slice_1:output:0*random_rotation_29/rotation_matrix/Neg:y:0;random_rotation_29/rotation_matrix/strided_slice_3:output:0;random_rotation_29/rotation_matrix/strided_slice_4:output:0;random_rotation_29/rotation_matrix/strided_slice_5:output:0;random_rotation_29/rotation_matrix/strided_slice_6:output:01random_rotation_29/rotation_matrix/zeros:output:07random_rotation_29/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"random_rotation_29/transform/ShapeShape4random_flip_32/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:z
0random_rotation_29/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2random_rotation_29/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2random_rotation_29/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
*random_rotation_29/transform/strided_sliceStridedSlice+random_rotation_29/transform/Shape:output:09random_rotation_29/transform/strided_slice/stack:output:0;random_rotation_29/transform/strided_slice/stack_1:output:0;random_rotation_29/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
'random_rotation_29/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
7random_rotation_29/transform/ImageProjectiveTransformV3ImageProjectiveTransformV34random_flip_32/stateless_random_flip_up_down/add:z:02random_rotation_29/rotation_matrix/concat:output:03random_rotation_29/transform/strided_slice:output:00random_rotation_29/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
random_zoom_32/ShapeShapeLrandom_rotation_29/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:l
"random_zoom_32/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$random_zoom_32/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$random_zoom_32/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom_32/strided_sliceStridedSlicerandom_zoom_32/Shape:output:0+random_zoom_32/strided_slice/stack:output:0-random_zoom_32/strided_slice/stack_1:output:0-random_zoom_32/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
$random_zoom_32/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿy
&random_zoom_32/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿp
&random_zoom_32/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
random_zoom_32/strided_slice_1StridedSlicerandom_zoom_32/Shape:output:0-random_zoom_32/strided_slice_1/stack:output:0/random_zoom_32/strided_slice_1/stack_1:output:0/random_zoom_32/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
random_zoom_32/CastCast'random_zoom_32/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: w
$random_zoom_32/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿy
&random_zoom_32/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿp
&random_zoom_32/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
random_zoom_32/strided_slice_2StridedSlicerandom_zoom_32/Shape:output:0-random_zoom_32/strided_slice_2/stack:output:0/random_zoom_32/strided_slice_2/stack_1:output:0/random_zoom_32/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
random_zoom_32/Cast_1Cast'random_zoom_32/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: i
'random_zoom_32/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
%random_zoom_32/stateful_uniform/shapePack%random_zoom_32/strided_slice:output:00random_zoom_32/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:h
#random_zoom_32/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?h
#random_zoom_32/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?o
%random_zoom_32/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ­
$random_zoom_32/stateful_uniform/ProdProd.random_zoom_32/stateful_uniform/shape:output:0.random_zoom_32/stateful_uniform/Const:output:0*
T0*
_output_shapes
: h
&random_zoom_32/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
&random_zoom_32/stateful_uniform/Cast_1Cast-random_zoom_32/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ò
.random_zoom_32/stateful_uniform/RngReadAndSkipRngReadAndSkip7random_zoom_32_stateful_uniform_rngreadandskip_resource/random_zoom_32/stateful_uniform/Cast/x:output:0*random_zoom_32/stateful_uniform/Cast_1:y:0*
_output_shapes
:}
3random_zoom_32/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5random_zoom_32/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5random_zoom_32/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:÷
-random_zoom_32/stateful_uniform/strided_sliceStridedSlice6random_zoom_32/stateful_uniform/RngReadAndSkip:value:0<random_zoom_32/stateful_uniform/strided_slice/stack:output:0>random_zoom_32/stateful_uniform/strided_slice/stack_1:output:0>random_zoom_32/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
'random_zoom_32/stateful_uniform/BitcastBitcast6random_zoom_32/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
5random_zoom_32/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7random_zoom_32/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7random_zoom_32/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:í
/random_zoom_32/stateful_uniform/strided_slice_1StridedSlice6random_zoom_32/stateful_uniform/RngReadAndSkip:value:0>random_zoom_32/stateful_uniform/strided_slice_1/stack:output:0@random_zoom_32/stateful_uniform/strided_slice_1/stack_1:output:0@random_zoom_32/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
)random_zoom_32/stateful_uniform/Bitcast_1Bitcast8random_zoom_32/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0~
<random_zoom_32/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ú
8random_zoom_32/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2.random_zoom_32/stateful_uniform/shape:output:02random_zoom_32/stateful_uniform/Bitcast_1:output:00random_zoom_32/stateful_uniform/Bitcast:output:0Erandom_zoom_32/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
#random_zoom_32/stateful_uniform/subSub,random_zoom_32/stateful_uniform/max:output:0,random_zoom_32/stateful_uniform/min:output:0*
T0*
_output_shapes
: È
#random_zoom_32/stateful_uniform/mulMulArandom_zoom_32/stateful_uniform/StatelessRandomUniformV2:output:0'random_zoom_32/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
random_zoom_32/stateful_uniformAddV2'random_zoom_32/stateful_uniform/mul:z:0,random_zoom_32/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
random_zoom_32/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
random_zoom_32/concatConcatV2#random_zoom_32/stateful_uniform:z:0#random_zoom_32/stateful_uniform:z:0#random_zoom_32/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
 random_zoom_32/zoom_matrix/ShapeShaperandom_zoom_32/concat:output:0*
T0*
_output_shapes
:x
.random_zoom_32/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0random_zoom_32/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0random_zoom_32/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
(random_zoom_32/zoom_matrix/strided_sliceStridedSlice)random_zoom_32/zoom_matrix/Shape:output:07random_zoom_32/zoom_matrix/strided_slice/stack:output:09random_zoom_32/zoom_matrix/strided_slice/stack_1:output:09random_zoom_32/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
 random_zoom_32/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom_32/zoom_matrix/subSubrandom_zoom_32/Cast_1:y:0)random_zoom_32/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: i
$random_zoom_32/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¡
"random_zoom_32/zoom_matrix/truedivRealDiv"random_zoom_32/zoom_matrix/sub:z:0-random_zoom_32/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
0random_zoom_32/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2random_zoom_32/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2random_zoom_32/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*random_zoom_32/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_32/concat:output:09random_zoom_32/zoom_matrix/strided_slice_1/stack:output:0;random_zoom_32/zoom_matrix/strided_slice_1/stack_1:output:0;random_zoom_32/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskg
"random_zoom_32/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?»
 random_zoom_32/zoom_matrix/sub_1Sub+random_zoom_32/zoom_matrix/sub_1/x:output:03random_zoom_32/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
random_zoom_32/zoom_matrix/mulMul&random_zoom_32/zoom_matrix/truediv:z:0$random_zoom_32/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"random_zoom_32/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
 random_zoom_32/zoom_matrix/sub_2Subrandom_zoom_32/Cast:y:0+random_zoom_32/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: k
&random_zoom_32/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
$random_zoom_32/zoom_matrix/truediv_1RealDiv$random_zoom_32/zoom_matrix/sub_2:z:0/random_zoom_32/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
0random_zoom_32/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
2random_zoom_32/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2random_zoom_32/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*random_zoom_32/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_32/concat:output:09random_zoom_32/zoom_matrix/strided_slice_2/stack:output:0;random_zoom_32/zoom_matrix/strided_slice_2/stack_1:output:0;random_zoom_32/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskg
"random_zoom_32/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?»
 random_zoom_32/zoom_matrix/sub_3Sub+random_zoom_32/zoom_matrix/sub_3/x:output:03random_zoom_32/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
 random_zoom_32/zoom_matrix/mul_1Mul(random_zoom_32/zoom_matrix/truediv_1:z:0$random_zoom_32/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0random_zoom_32/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
2random_zoom_32/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2random_zoom_32/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*random_zoom_32/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_32/concat:output:09random_zoom_32/zoom_matrix/strided_slice_3/stack:output:0;random_zoom_32/zoom_matrix/strided_slice_3/stack_1:output:0;random_zoom_32/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskk
)random_zoom_32/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ä
'random_zoom_32/zoom_matrix/zeros/packedPack1random_zoom_32/zoom_matrix/strided_slice:output:02random_zoom_32/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&random_zoom_32/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ½
 random_zoom_32/zoom_matrix/zerosFill0random_zoom_32/zoom_matrix/zeros/packed:output:0/random_zoom_32/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+random_zoom_32/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :È
)random_zoom_32/zoom_matrix/zeros_1/packedPack1random_zoom_32/zoom_matrix/strided_slice:output:04random_zoom_32/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(random_zoom_32/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
"random_zoom_32/zoom_matrix/zeros_1Fill2random_zoom_32/zoom_matrix/zeros_1/packed:output:01random_zoom_32/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0random_zoom_32/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
2random_zoom_32/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
2random_zoom_32/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
*random_zoom_32/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_32/concat:output:09random_zoom_32/zoom_matrix/strided_slice_4/stack:output:0;random_zoom_32/zoom_matrix/strided_slice_4/stack_1:output:0;random_zoom_32/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskm
+random_zoom_32/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :È
)random_zoom_32/zoom_matrix/zeros_2/packedPack1random_zoom_32/zoom_matrix/strided_slice:output:04random_zoom_32/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:m
(random_zoom_32/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ã
"random_zoom_32/zoom_matrix/zeros_2Fill2random_zoom_32/zoom_matrix/zeros_2/packed:output:01random_zoom_32/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&random_zoom_32/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ò
!random_zoom_32/zoom_matrix/concatConcatV23random_zoom_32/zoom_matrix/strided_slice_3:output:0)random_zoom_32/zoom_matrix/zeros:output:0"random_zoom_32/zoom_matrix/mul:z:0+random_zoom_32/zoom_matrix/zeros_1:output:03random_zoom_32/zoom_matrix/strided_slice_4:output:0$random_zoom_32/zoom_matrix/mul_1:z:0+random_zoom_32/zoom_matrix/zeros_2:output:0/random_zoom_32/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
random_zoom_32/transform/ShapeShapeLrandom_rotation_29/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:v
,random_zoom_32/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:x
.random_zoom_32/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.random_zoom_32/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
&random_zoom_32/transform/strided_sliceStridedSlice'random_zoom_32/transform/Shape:output:05random_zoom_32/transform/strided_slice/stack:output:07random_zoom_32/transform/strided_slice/stack_1:output:07random_zoom_32/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:h
#random_zoom_32/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
3random_zoom_32/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Lrandom_rotation_29/transform/ImageProjectiveTransformV3:transformed_images:0*random_zoom_32/zoom_matrix/concat:output:0/random_zoom_32/transform/strided_slice:output:0,random_zoom_32/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR¡
IdentityIdentityHrandom_zoom_32/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
NoOpNoOp8^random_flip_32/stateful_uniform_full_int/RngReadAndSkip:^random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip3^random_rotation_29/stateful_uniform/RngReadAndSkip/^random_zoom_32/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : 2r
7random_flip_32/stateful_uniform_full_int/RngReadAndSkip7random_flip_32/stateful_uniform_full_int/RngReadAndSkip2v
9random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip9random_flip_32/stateful_uniform_full_int_1/RngReadAndSkip2h
2random_rotation_29/stateful_uniform/RngReadAndSkip2random_rotation_29/stateful_uniform/RngReadAndSkip2`
.random_zoom_32/stateful_uniform/RngReadAndSkip.random_zoom_32/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
n
Â
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_77258

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_76737

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø&
£
H__inference_sequential_64_layer_call_and_return_conditional_losses_75938

inputsB
(conv2d_62_conv2d_readvariableop_resource:
7
)conv2d_62_biasadd_readvariableop_resource:
B
(conv2d_63_conv2d_readvariableop_resource:

7
)conv2d_63_biasadd_readvariableop_resource:
;
'dense_31_matmul_readvariableop_resource:
¨¬6
(dense_31_biasadd_readvariableop_resource:
identity¢ conv2d_62/BiasAdd/ReadVariableOp¢conv2d_62/Conv2D/ReadVariableOp¢ conv2d_63/BiasAdd/ReadVariableOp¢conv2d_63/Conv2D/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢dense_31/MatMul/ReadVariableOp
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0°
conv2d_62/Conv2DConv2Dinputs'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
*
paddingVALID*
strides

 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
n
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
z
dropout_116/IdentityIdentityconv2d_62/Relu:activations:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ
¯
max_pooling2d_62/MaxPoolMaxPooldropout_116/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
}
dropout_117/IdentityIdentity!max_pooling2d_62/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0Å
conv2d_63/Conv2DConv2Ddropout_117/Identity:output:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
*
paddingVALID*
strides

 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
l
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
x
dropout_118/IdentityIdentityconv2d_63/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}
¯
max_pooling2d_63/MaxPoolMaxPooldropout_118/Identity:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
*
ksize
*
paddingVALID*
strides
}
dropout_119/IdentityIdentity!max_pooling2d_63/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>
a
flatten_28/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(  
flatten_28/ReshapeReshapedropout_119/Identity:output:0flatten_28/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¬
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
¨¬*
dtype0
dense_31/MatMulMatMulflatten_28/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitydense_31/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
]
sequential_63_inputF
%serving_default_sequential_63_input:0ÿÿÿÿÿÿÿÿÿ<
dense_310
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
Ä
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-2
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ñ
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
»

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*_random_generator
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7_random_generator
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
»

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F_random_generator
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S_random_generator
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
»

\kernel
]bias
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
Ë
diter

ebeta_1

fbeta_2
	gdecay
hlearning_ratemÛmÜ:mÝ;mÞ\mß]màvávâ:vã;vä\vå]væ"
	optimizer
J
0
1
:2
;3
\4
]5"
trackable_list_wrapper
J
0
1
:2
;3
\4
]5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_64_layer_call_fn_75575
-__inference_sequential_64_layer_call_fn_75882
-__inference_sequential_64_layer_call_fn_75905
-__inference_sequential_64_layer_call_fn_75799À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_64_layer_call_and_return_conditional_losses_75938
H__inference_sequential_64_layer_call_and_return_conditional_losses_76318
H__inference_sequential_64_layer_call_and_return_conditional_losses_75826
H__inference_sequential_64_layer_call_and_return_conditional_losses_75859À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×BÔ
 __inference__wrapped_model_74979sequential_63_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
nserving_default"
signature_map
¼
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s_random_generator
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z_random_generator
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
À
}	variables
~trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_63_layer_call_fn_75008
-__inference_sequential_63_layer_call_fn_76342
-__inference_sequential_63_layer_call_fn_76353
-__inference_sequential_63_layer_call_fn_75418À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_63_layer_call_and_return_conditional_losses_76357
H__inference_sequential_63_layer_call_and_return_conditional_losses_76680
H__inference_sequential_63_layer_call_and_return_conditional_losses_75425
H__inference_sequential_63_layer_call_and_return_conditional_losses_75438À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
*:(
2conv2d_62/kernel
:
2conv2d_62/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_62_layer_call_fn_76689¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_62_layer_call_and_return_conditional_losses_76700¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
&	variables
'trainable_variables
(regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_116_layer_call_fn_76705
+__inference_dropout_116_layer_call_fn_76710´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_116_layer_call_and_return_conditional_losses_76715
F__inference_dropout_116_layer_call_and_return_conditional_losses_76727´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_62_layer_call_fn_76732¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_76737¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_117_layer_call_fn_76742
+__inference_dropout_117_layer_call_fn_76747´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_117_layer_call_and_return_conditional_losses_76752
F__inference_dropout_117_layer_call_and_return_conditional_losses_76764´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
*:(

2conv2d_63/kernel
:
2conv2d_63/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
  layer_regularization_losses
¡layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_63_layer_call_fn_76773¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_63_layer_call_and_return_conditional_losses_76784¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¢non_trainable_variables
£layers
¤metrics
 ¥layer_regularization_losses
¦layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_118_layer_call_fn_76789
+__inference_dropout_118_layer_call_fn_76794´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_118_layer_call_and_return_conditional_losses_76799
F__inference_dropout_118_layer_call_and_return_conditional_losses_76811´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_63_layer_call_fn_76816¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_76821¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
+__inference_dropout_119_layer_call_fn_76826
+__inference_dropout_119_layer_call_fn_76831´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_119_layer_call_and_return_conditional_losses_76836
F__inference_dropout_119_layer_call_and_return_conditional_losses_76848´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_flatten_28_layer_call_fn_76853¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_flatten_28_layer_call_and_return_conditional_losses_76859¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
¨¬2dense_31/kernel
:2dense_31/bias
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_31_layer_call_fn_76868¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_31_layer_call_and_return_conditional_losses_76879¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
»0
¼1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÖBÓ
#__inference_signature_wrapper_76337sequential_63_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
o	variables
ptrainable_variables
qregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
/
Â
_generator"
_generic_user_object
2
.__inference_random_flip_32_layer_call_fn_76884
.__inference_random_flip_32_layer_call_fn_76891´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_random_flip_32_layer_call_and_return_conditional_losses_76895
I__inference_random_flip_32_layer_call_and_return_conditional_losses_77006´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
/
È
_generator"
_generic_user_object
¢2
2__inference_random_rotation_29_layer_call_fn_77011
2__inference_random_rotation_29_layer_call_fn_77018´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_77022
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_77140´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
}	variables
~trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
Î
_generator"
_generic_user_object
2
.__inference_random_zoom_32_layer_call_fn_77145
.__inference_random_zoom_32_layer_call_fn_77152´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_77156
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_77258´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

Ïtotal

Ðcount
Ñ	variables
Ò	keras_api"
_tf_keras_metric
c

Ótotal

Ôcount
Õ
_fn_kwargs
Ö	variables
×	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
Ø
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
Ù
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
/
Ú
_state_var"
_generic_user_object
:  (2total
:  (2count
0
Ï0
Ð1"
trackable_list_wrapper
.
Ñ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ó0
Ô1"
trackable_list_wrapper
.
Ö	variables"
_generic_user_object
?:=	23sequential_64/sequential_63/random_flip_32/StateVar
C:A	27sequential_64/sequential_63/random_rotation_29/StateVar
?:=	23sequential_64/sequential_63/random_zoom_32/StateVar
/:-
2Adam/conv2d_62/kernel/m
!:
2Adam/conv2d_62/bias/m
/:-

2Adam/conv2d_63/kernel/m
!:
2Adam/conv2d_63/bias/m
(:&
¨¬2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
/:-
2Adam/conv2d_62/kernel/v
!:
2Adam/conv2d_62/bias/v
/:-

2Adam/conv2d_63/kernel/v
!:
2Adam/conv2d_63/bias/v
(:&
¨¬2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/vª
 __inference__wrapped_model_74979:;\]F¢C
<¢9
74
sequential_63_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_31"
dense_31ÿÿÿÿÿÿÿÿÿ¸
D__inference_conv2d_62_layer_call_and_return_conditional_losses_76700p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿþþ

 
)__inference_conv2d_62_layer_call_fn_76689c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿþþ
´
D__inference_conv2d_63_layer_call_and_return_conditional_losses_76784l:;7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ}}

 
)__inference_conv2d_63_layer_call_fn_76773_:;7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª " ÿÿÿÿÿÿÿÿÿ}}
¥
C__inference_dense_31_layer_call_and_return_conditional_losses_76879^\]1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ¨¬
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_31_layer_call_fn_76868Q\]1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ¨¬
ª "ÿÿÿÿÿÿÿÿÿº
F__inference_dropout_116_layer_call_and_return_conditional_losses_76715p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿþþ

p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿþþ

 º
F__inference_dropout_116_layer_call_and_return_conditional_losses_76727p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿþþ

p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿþþ

 
+__inference_dropout_116_layer_call_fn_76705c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿþþ

p 
ª ""ÿÿÿÿÿÿÿÿÿþþ

+__inference_dropout_116_layer_call_fn_76710c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿþþ

p
ª ""ÿÿÿÿÿÿÿÿÿþþ
¶
F__inference_dropout_117_layer_call_and_return_conditional_losses_76752l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 ¶
F__inference_dropout_117_layer_call_and_return_conditional_losses_76764l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
+__inference_dropout_117_layer_call_fn_76742_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p 
ª " ÿÿÿÿÿÿÿÿÿ

+__inference_dropout_117_layer_call_fn_76747_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p
ª " ÿÿÿÿÿÿÿÿÿ
¶
F__inference_dropout_118_layer_call_and_return_conditional_losses_76799l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ}}

p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ}}

 ¶
F__inference_dropout_118_layer_call_and_return_conditional_losses_76811l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ}}

p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ}}

 
+__inference_dropout_118_layer_call_fn_76789_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ}}

p 
ª " ÿÿÿÿÿÿÿÿÿ}}

+__inference_dropout_118_layer_call_fn_76794_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ}}

p
ª " ÿÿÿÿÿÿÿÿÿ}}
¶
F__inference_dropout_119_layer_call_and_return_conditional_losses_76836l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>

p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ>>

 ¶
F__inference_dropout_119_layer_call_and_return_conditional_losses_76848l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>

p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ>>

 
+__inference_dropout_119_layer_call_fn_76826_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>

p 
ª " ÿÿÿÿÿÿÿÿÿ>>

+__inference_dropout_119_layer_call_fn_76831_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>

p
ª " ÿÿÿÿÿÿÿÿÿ>>
«
E__inference_flatten_28_layer_call_and_return_conditional_losses_76859b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ>>

ª "'¢$

0ÿÿÿÿÿÿÿÿÿ¨¬
 
*__inference_flatten_28_layer_call_fn_76853U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ>>

ª "ÿÿÿÿÿÿÿÿÿ¨¬î
K__inference_max_pooling2d_62_layer_call_and_return_conditional_losses_76737R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_62_layer_call_fn_76732R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_63_layer_call_and_return_conditional_losses_76821R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_63_layer_call_fn_76816R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
I__inference_random_flip_32_layer_call_and_return_conditional_losses_76895p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Á
I__inference_random_flip_32_layer_call_and_return_conditional_losses_77006tØ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_random_flip_32_layer_call_fn_76884c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
.__inference_random_flip_32_layer_call_fn_76891gØ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿÁ
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_77022p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Å
M__inference_random_rotation_29_layer_call_and_return_conditional_losses_77140tÙ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
2__inference_random_rotation_29_layer_call_fn_77011c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
2__inference_random_rotation_29_layer_call_fn_77018gÙ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ½
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_77156p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Á
I__inference_random_zoom_32_layer_call_and_return_conditional_losses_77258tÚ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_random_zoom_32_layer_call_fn_77145c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
.__inference_random_zoom_32_layer_call_fn_77152gÚ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿÏ
H__inference_sequential_63_layer_call_and_return_conditional_losses_75425O¢L
E¢B
85
random_flip_32_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ×
H__inference_sequential_63_layer_call_and_return_conditional_losses_75438ØÙÚO¢L
E¢B
85
random_flip_32_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 À
H__inference_sequential_63_layer_call_and_return_conditional_losses_76357tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 È
H__inference_sequential_63_layer_call_and_return_conditional_losses_76680|ØÙÚA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¦
-__inference_sequential_63_layer_call_fn_75008uO¢L
E¢B
85
random_flip_32_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ®
-__inference_sequential_63_layer_call_fn_75418}ØÙÚO¢L
E¢B
85
random_flip_32_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_63_layer_call_fn_76342gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ 
-__inference_sequential_63_layer_call_fn_76353oØÙÚA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿË
H__inference_sequential_64_layer_call_and_return_conditional_losses_75826:;\]N¢K
D¢A
74
sequential_63_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
H__inference_sequential_64_layer_call_and_return_conditional_losses_75859ØÙÚ:;\]N¢K
D¢A
74
sequential_63_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
H__inference_sequential_64_layer_call_and_return_conditional_losses_75938r:;\]A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_sequential_64_layer_call_and_return_conditional_losses_76318xØÙÚ:;\]A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
-__inference_sequential_64_layer_call_fn_75575r:;\]N¢K
D¢A
74
sequential_63_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
-__inference_sequential_64_layer_call_fn_75799xØÙÚ:;\]N¢K
D¢A
74
sequential_63_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_64_layer_call_fn_75882e:;\]A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_64_layer_call_fn_75905kØÙÚ:;\]A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÄ
#__inference_signature_wrapper_76337:;\]]¢Z
¢ 
SªP
N
sequential_63_input74
sequential_63_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_31"
dense_31ÿÿÿÿÿÿÿÿÿ