´Ů
ö.Ć.
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
2
NextIteration	
data"T
output"T"	
Ttype
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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

ReverseSequence

input"T
seq_lengths"Tlen
output"T"
seq_dimint"
	batch_dimint "	
Ttype"
Tlentype0	:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Ţ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.13.12b'v1.13.0-rc2-5-g6612da8951'

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container 
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step

encoded_textPlaceholder*
dtype0	*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
b
lengthsPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
random_normal/shapeConst*
valueB"J      *
dtype0*
_output_shapes
:
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
T0*
dtype0*
_output_shapes

:J*
seed2 *

seed 
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes

:J
d
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes

:J
~

embeddings
VariableV2*
shape
:J*
shared_name *
dtype0*
_output_shapes

:J*
	container 
§
embeddings/AssignAssign
embeddingsrandom_normal*
T0*
_class
loc:@embeddings*
validate_shape(*
_output_shapes

:J*
use_locking(
o
embeddings/readIdentity
embeddings*
T0*
_class
loc:@embeddings*
_output_shapes

:J
v
embedding_lookup/axisConst*
dtype0*
_output_shapes
: *
value	B : *
_class
loc:@embeddings
Ó
embedding_lookupGatherV2embeddings/readencoded_textembedding_lookup/axis*
Tparams0*
_class
loc:@embeddings*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Taxis0*
Tindices0	
v
embedding_lookup/IdentityIdentityembedding_lookup*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
]
DropoutWrapperInit/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
_
DropoutWrapperInit_1/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_1/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_1/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_2/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_2/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_2/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
_
DropoutWrapperInit_3/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_3/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_3/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_4/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_4/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_4/Const_2Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit_5/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
a
DropoutWrapperInit_5/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
DropoutWrapperInit_5/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
^
bidirectional_rnn/fw/fw/RankConst*
dtype0*
_output_shapes
: *
value	B :
e
#bidirectional_rnn/fw/fw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
e
#bidirectional_rnn/fw/fw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ś
bidirectional_rnn/fw/fw/rangeRange#bidirectional_rnn/fw/fw/range/startbidirectional_rnn/fw/fw/Rank#bidirectional_rnn/fw/fw/range/delta*
_output_shapes
:*

Tidx0
x
'bidirectional_rnn/fw/fw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
e
#bidirectional_rnn/fw/fw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ń
bidirectional_rnn/fw/fw/concatConcatV2'bidirectional_rnn/fw/fw/concat/values_0bidirectional_rnn/fw/fw/range#bidirectional_rnn/fw/fw/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
ľ
!bidirectional_rnn/fw/fw/transpose	Transposeembedding_lookup/Identitybidirectional_rnn/fw/fw/concat*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tperm0
}
bidirectional_rnn/fw/fw/ToInt32Castlengths*

SrcT0	*
Truncate( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

'bidirectional_rnn/fw/fw/sequence_lengthIdentitybidirectional_rnn/fw/fw/ToInt32*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
bidirectional_rnn/fw/fw/ShapeShape!bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
u
+bidirectional_rnn/fw/fw/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
w
-bidirectional_rnn/fw/fw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-bidirectional_rnn/fw/fw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ń
%bidirectional_rnn/fw/fw/strided_sliceStridedSlicebidirectional_rnn/fw/fw/Shape+bidirectional_rnn/fw/fw/strided_slice/stack-bidirectional_rnn/fw/fw/strided_slice/stack_1-bidirectional_rnn/fw/fw/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ś
tbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ě
pbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims
ExpandDims%bidirectional_rnn/fw/fw/strided_slicetbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
ś
kbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
qbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

lbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concatConcatV2pbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDimskbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/Constqbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
ś
qbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

kbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zerosFilllbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concatqbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¸
vbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Đ
rbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims_1
ExpandDims%bidirectional_rnn/fw/fw/strided_slicevbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
¸
mbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
¸
vbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Đ
rbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims
ExpandDims%bidirectional_rnn/fw/fw/strided_slicevbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
¸
mbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
ľ
sbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

nbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concatConcatV2rbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDimsmbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/Constsbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
¸
sbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

mbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zerosFillnbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concatsbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
xbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ô
tbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims_1
ExpandDims%bidirectional_rnn/fw/fw/strided_slicexbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
ş
obidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
¸
vbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Đ
rbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims
ExpandDims%bidirectional_rnn/fw/fw/strided_slicevbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
¸
mbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:
ľ
sbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

nbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concatConcatV2rbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDimsmbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/Constsbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
¸
sbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

mbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zerosFillnbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concatsbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
xbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ô
tbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims_1
ExpandDims%bidirectional_rnn/fw/fw/strided_slicexbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
ş
obidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional_rnn/fw/fw/Shape_1Shape'bidirectional_rnn/fw/fw/sequence_length*
T0*
out_type0*
_output_shapes
:

bidirectional_rnn/fw/fw/stackPack%bidirectional_rnn/fw/fw/strided_slice*
T0*

axis *
N*
_output_shapes
:

bidirectional_rnn/fw/fw/EqualEqualbidirectional_rnn/fw/fw/Shape_1bidirectional_rnn/fw/fw/stack*
T0*
_output_shapes
:
g
bidirectional_rnn/fw/fw/ConstConst*
valueB: *
dtype0*
_output_shapes
:

bidirectional_rnn/fw/fw/AllAllbidirectional_rnn/fw/fw/Equalbidirectional_rnn/fw/fw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
Ź
$bidirectional_rnn/fw/fw/Assert/ConstConst*
dtype0*
_output_shapes
: *X
valueOBM BGExpected shape for Tensor bidirectional_rnn/fw/fw/sequence_length:0 is 
w
&bidirectional_rnn/fw/fw/Assert/Const_1Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 
´
,bidirectional_rnn/fw/fw/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *X
valueOBM BGExpected shape for Tensor bidirectional_rnn/fw/fw/sequence_length:0 is 
}
,bidirectional_rnn/fw/fw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

%bidirectional_rnn/fw/fw/Assert/AssertAssertbidirectional_rnn/fw/fw/All,bidirectional_rnn/fw/fw/Assert/Assert/data_0bidirectional_rnn/fw/fw/stack,bidirectional_rnn/fw/fw/Assert/Assert/data_2bidirectional_rnn/fw/fw/Shape_1*
T
2*
	summarize
Ž
#bidirectional_rnn/fw/fw/CheckSeqLenIdentity'bidirectional_rnn/fw/fw/sequence_length&^bidirectional_rnn/fw/fw/Assert/Assert*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

bidirectional_rnn/fw/fw/Shape_2Shape!bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:
w
-bidirectional_rnn/fw/fw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/bidirectional_rnn/fw/fw/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
y
/bidirectional_rnn/fw/fw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ű
'bidirectional_rnn/fw/fw/strided_slice_1StridedSlicebidirectional_rnn/fw/fw/Shape_2-bidirectional_rnn/fw/fw/strided_slice_1/stack/bidirectional_rnn/fw/fw/strided_slice_1/stack_1/bidirectional_rnn/fw/fw/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

bidirectional_rnn/fw/fw/Shape_3Shape!bidirectional_rnn/fw/fw/transpose*
_output_shapes
:*
T0*
out_type0
w
-bidirectional_rnn/fw/fw/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
y
/bidirectional_rnn/fw/fw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/bidirectional_rnn/fw/fw/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ű
'bidirectional_rnn/fw/fw/strided_slice_2StridedSlicebidirectional_rnn/fw/fw/Shape_3-bidirectional_rnn/fw/fw/strided_slice_2/stack/bidirectional_rnn/fw/fw/strided_slice_2/stack_1/bidirectional_rnn/fw/fw/strided_slice_2/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
h
&bidirectional_rnn/fw/fw/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
˛
"bidirectional_rnn/fw/fw/ExpandDims
ExpandDims'bidirectional_rnn/fw/fw/strided_slice_2&bidirectional_rnn/fw/fw/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
j
bidirectional_rnn/fw/fw/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
g
%bidirectional_rnn/fw/fw/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ň
 bidirectional_rnn/fw/fw/concat_1ConcatV2"bidirectional_rnn/fw/fw/ExpandDimsbidirectional_rnn/fw/fw/Const_1%bidirectional_rnn/fw/fw/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
h
#bidirectional_rnn/fw/fw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
bidirectional_rnn/fw/fw/zerosFill bidirectional_rnn/fw/fw/concat_1#bidirectional_rnn/fw/fw/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
bidirectional_rnn/fw/fw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Ś
bidirectional_rnn/fw/fw/MinMin#bidirectional_rnn/fw/fw/CheckSeqLenbidirectional_rnn/fw/fw/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
bidirectional_rnn/fw/fw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
Ś
bidirectional_rnn/fw/fw/MaxMax#bidirectional_rnn/fw/fw/CheckSeqLenbidirectional_rnn/fw/fw/Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
^
bidirectional_rnn/fw/fw/timeConst*
value	B : *
dtype0*
_output_shapes
: 
Ŕ
#bidirectional_rnn/fw/fw/TensorArrayTensorArrayV3'bidirectional_rnn/fw/fw/strided_slice_1*C
tensor_array_name.,bidirectional_rnn/fw/fw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:˙˙˙˙˙˙˙˙˙*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
Ŕ
%bidirectional_rnn/fw/fw/TensorArray_1TensorArrayV3'bidirectional_rnn/fw/fw/strided_slice_1*B
tensor_array_name-+bidirectional_rnn/fw/fw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:˙˙˙˙˙˙˙˙˙*
clear_after_read(*
dynamic_size( *
identical_element_shapes(

0bidirectional_rnn/fw/fw/TensorArrayUnstack/ShapeShape!bidirectional_rnn/fw/fw/transpose*
T0*
out_type0*
_output_shapes
:

>bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
Đ
8bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_sliceStridedSlice0bidirectional_rnn/fw/fw/TensorArrayUnstack/Shape>bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_1@bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
x
6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

0bidirectional_rnn/fw/fw/TensorArrayUnstack/rangeRange6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/start8bidirectional_rnn/fw/fw/TensorArrayUnstack/strided_slice6bidirectional_rnn/fw/fw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ć
Rbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3%bidirectional_rnn/fw/fw/TensorArray_10bidirectional_rnn/fw/fw/TensorArrayUnstack/range!bidirectional_rnn/fw/fw/transpose'bidirectional_rnn/fw/fw/TensorArray_1:1*
_output_shapes
: *
T0*4
_class*
(&loc:@bidirectional_rnn/fw/fw/transpose
c
!bidirectional_rnn/fw/fw/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :

bidirectional_rnn/fw/fw/MaximumMaximum!bidirectional_rnn/fw/fw/Maximum/xbidirectional_rnn/fw/fw/Max*
T0*
_output_shapes
: 

bidirectional_rnn/fw/fw/MinimumMinimum'bidirectional_rnn/fw/fw/strided_slice_1bidirectional_rnn/fw/fw/Maximum*
_output_shapes
: *
T0
q
/bidirectional_rnn/fw/fw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
é
#bidirectional_rnn/fw/fw/while/EnterEnter/bidirectional_rnn/fw/fw/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ř
%bidirectional_rnn/fw/fw/while/Enter_1Enterbidirectional_rnn/fw/fw/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
á
%bidirectional_rnn/fw/fw/while/Enter_2Enter%bidirectional_rnn/fw/fw/TensorArray:1*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant( 
š
%bidirectional_rnn/fw/fw/while/Enter_3Enterkbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ť
%bidirectional_rnn/fw/fw/while/Enter_4Entermbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ť
%bidirectional_rnn/fw/fw/while/Enter_5Entermbidirectional_rnn/fw/fw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ş
#bidirectional_rnn/fw/fw/while/MergeMerge#bidirectional_rnn/fw/fw/while/Enter+bidirectional_rnn/fw/fw/while/NextIteration*
T0*
N*
_output_shapes
: : 
°
%bidirectional_rnn/fw/fw/while/Merge_1Merge%bidirectional_rnn/fw/fw/while/Enter_1-bidirectional_rnn/fw/fw/while/NextIteration_1*
N*
_output_shapes
: : *
T0
°
%bidirectional_rnn/fw/fw/while/Merge_2Merge%bidirectional_rnn/fw/fw/while/Enter_2-bidirectional_rnn/fw/fw/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
Â
%bidirectional_rnn/fw/fw/while/Merge_3Merge%bidirectional_rnn/fw/fw/while/Enter_3-bidirectional_rnn/fw/fw/while/NextIteration_3*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Â
%bidirectional_rnn/fw/fw/while/Merge_4Merge%bidirectional_rnn/fw/fw/while/Enter_4-bidirectional_rnn/fw/fw/while/NextIteration_4*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Â
%bidirectional_rnn/fw/fw/while/Merge_5Merge%bidirectional_rnn/fw/fw/while/Enter_5-bidirectional_rnn/fw/fw/while/NextIteration_5*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

"bidirectional_rnn/fw/fw/while/LessLess#bidirectional_rnn/fw/fw/while/Merge(bidirectional_rnn/fw/fw/while/Less/Enter*
T0*
_output_shapes
: 
ć
(bidirectional_rnn/fw/fw/while/Less/EnterEnter'bidirectional_rnn/fw/fw/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
 
$bidirectional_rnn/fw/fw/while/Less_1Less%bidirectional_rnn/fw/fw/while/Merge_1*bidirectional_rnn/fw/fw/while/Less_1/Enter*
T0*
_output_shapes
: 
ŕ
*bidirectional_rnn/fw/fw/while/Less_1/EnterEnterbidirectional_rnn/fw/fw/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context

(bidirectional_rnn/fw/fw/while/LogicalAnd
LogicalAnd"bidirectional_rnn/fw/fw/while/Less$bidirectional_rnn/fw/fw/while/Less_1*
_output_shapes
: 
t
&bidirectional_rnn/fw/fw/while/LoopCondLoopCond(bidirectional_rnn/fw/fw/while/LogicalAnd*
_output_shapes
: 
Ö
$bidirectional_rnn/fw/fw/while/SwitchSwitch#bidirectional_rnn/fw/fw/while/Merge&bidirectional_rnn/fw/fw/while/LoopCond*
T0*6
_class,
*(loc:@bidirectional_rnn/fw/fw/while/Merge*
_output_shapes
: : 
Ü
&bidirectional_rnn/fw/fw/while/Switch_1Switch%bidirectional_rnn/fw/fw/while/Merge_1&bidirectional_rnn/fw/fw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_1*
_output_shapes
: : 
Ü
&bidirectional_rnn/fw/fw/while/Switch_2Switch%bidirectional_rnn/fw/fw/while/Merge_2&bidirectional_rnn/fw/fw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_2*
_output_shapes
: : 

&bidirectional_rnn/fw/fw/while/Switch_3Switch%bidirectional_rnn/fw/fw/while/Merge_3&bidirectional_rnn/fw/fw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_3*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

&bidirectional_rnn/fw/fw/while/Switch_4Switch%bidirectional_rnn/fw/fw/while/Merge_4&bidirectional_rnn/fw/fw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_4*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

&bidirectional_rnn/fw/fw/while/Switch_5Switch%bidirectional_rnn/fw/fw/while/Merge_5&bidirectional_rnn/fw/fw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/fw/fw/while/Merge_5*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
{
&bidirectional_rnn/fw/fw/while/IdentityIdentity&bidirectional_rnn/fw/fw/while/Switch:1*
T0*
_output_shapes
: 

(bidirectional_rnn/fw/fw/while/Identity_1Identity(bidirectional_rnn/fw/fw/while/Switch_1:1*
T0*
_output_shapes
: 

(bidirectional_rnn/fw/fw/while/Identity_2Identity(bidirectional_rnn/fw/fw/while/Switch_2:1*
T0*
_output_shapes
: 

(bidirectional_rnn/fw/fw/while/Identity_3Identity(bidirectional_rnn/fw/fw/while/Switch_3:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

(bidirectional_rnn/fw/fw/while/Identity_4Identity(bidirectional_rnn/fw/fw/while/Switch_4:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

(bidirectional_rnn/fw/fw/while/Identity_5Identity(bidirectional_rnn/fw/fw/while/Switch_5:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

#bidirectional_rnn/fw/fw/while/add/yConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :

!bidirectional_rnn/fw/fw/while/addAdd&bidirectional_rnn/fw/fw/while/Identity#bidirectional_rnn/fw/fw/while/add/y*
T0*
_output_shapes
: 

/bidirectional_rnn/fw/fw/while/TensorArrayReadV3TensorArrayReadV35bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter(bidirectional_rnn/fw/fw/while/Identity_17bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
5bidirectional_rnn/fw/fw/while/TensorArrayReadV3/EnterEnter%bidirectional_rnn/fw/fw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
 
7bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1EnterRbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
Ä
*bidirectional_rnn/fw/fw/while/GreaterEqualGreaterEqual(bidirectional_rnn/fw/fw/while/Identity_10bidirectional_rnn/fw/fw/while/GreaterEqual/Enter*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
÷
0bidirectional_rnn/fw/fw/while/GreaterEqual/EnterEnter#bidirectional_rnn/fw/fw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Š
rbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*
valueB"      *d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
:

pbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/minConst*
valueB
 *Ń_ý˝*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 

pbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ń_ý=*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 
ś
zbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniformrbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shape*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
â
pbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/subSubpbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxpbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel
ö
pbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulMulzbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformpbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel
č
lbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniformAddpbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulpbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

÷
Qbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelVarHandleOp*
shape:
*
dtype0*
_output_shapes
: *b
shared_nameSQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
	container 
ó
rbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
 
Xbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/AssignAssignVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernellbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0
ß
ebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpReadVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0* 
_output_shapes
:
*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel
ý
_bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/IdentityIdentityebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp* 
_output_shapes
:
*
T0

abidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/ConstConst*
dtype0*
_output_shapes	
:*
valueB*  ?*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias
ě
Obidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *`
shared_nameQObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias
ď
pbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
_output_shapes
: 

Vbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/AssignAssignVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasabidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/Const*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
dtype0
Ô
cbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpReadVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ô
]bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/IdentityIdentitycbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ô
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Ć
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *AWž*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
Ć
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *AW>*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
ö
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
seed2 
š
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Ě
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/sub*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
:	
ž
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
:	
ľ
fbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelVarHandleOp*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: *w
shared_namehfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel

bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
ő
mbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/AssignAssignVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform*
dtype0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel

zbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpReadVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
:	
Ś
tbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/IdentityIdentityzbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOp*
T0*
_output_shapes
:	
Ö
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
:
Č
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *qÄž*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Č
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
ú
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
seed2 
˝
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
Ń
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Ă
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

š
gbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelVarHandleOp*
dtype0*
_output_shapes
: *x
shared_nameigbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
	container *
shape:

 
bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpgbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
ů
nbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/AssignAssignVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0
Ą
{bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpReadVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0* 
_output_shapes
:

Š
ubidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/IdentityIdentity{bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOp* 
_output_shapes
:
*
T0
ž
vbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zerosConst*
valueB*    *w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:
Ť
dbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasVarHandleOp*
dtype0*
_output_shapes
: *u
shared_namefdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
	container *
shape:

bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
_output_shapes
: 
ă
kbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/AssignAssignVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasvbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros*
dtype0*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias

xbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpReadVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:

rbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/IdentityIdentityxbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ŕ
wbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias
Ž
ebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biasVarHandleOp*v
shared_namegebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
	container *
shape:*
dtype0*
_output_shapes
: 

bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
_output_shapes
: 
ç
lbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/AssignAssignVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biaswbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0

ybidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOpReadVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias
 
sbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/IdentityIdentityybidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ç
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat/axisConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concatConcatV2/bidirectional_rnn/fw/fw/while/TensorArrayReadV3(bidirectional_rnn/fw/fw/while/Identity_3\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat/axis*

Tidx0*
T0*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ň
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMulMatMulWbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat]bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Ý
]bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/EnterEnter_bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ć
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAddBiasAddWbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/EnterEnter]bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
đ
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/SigmoidSigmoidXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/ConstConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ë
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split/split_dimConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
ó
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/splitSplit`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split/split_dimXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Sigmoid*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split
Î
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1MatMul/bidirectional_rnn/fw/fw/while/TensorArrayReadV3_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ó
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/EnterEntertbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:	*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ě
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1BiasAddYbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/EnterEnterrbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ç
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2MatMul(bidirectional_rnn/fw/fw/while/Identity_3_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ő
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/EnterEnterubidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ě
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2BiasAddYbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/EnterEntersbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Â
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mulMulVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/splitZbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/addAddZbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Ubidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/TanhTanhTbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub/xConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ŕ
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/subSubVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub/xXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_1MulTbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/subUbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_2MulXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:1(bidirectional_rnn/fw/fw/while/Identity_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1AddVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_1Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
rbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*
valueB"      *d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
:

pbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 

pbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 
ś
zbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniformrbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
seed2 
â
pbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/subSubpbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxpbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
ö
pbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulMulzbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformpbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/sub*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

č
lbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniformAddpbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulpbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

÷
Qbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelVarHandleOp*
shape:
*
dtype0*
_output_shapes
: *b
shared_nameSQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
	container 
ó
rbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
 
Xbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/AssignAssignVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernellbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0
ß
ebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpReadVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0* 
_output_shapes
:

ý
_bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/IdentityIdentityebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:


abidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/ConstConst*
valueB*  ?*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ě
Obidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *`
shared_nameQObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
	container 
ď
pbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
_output_shapes
: 

Vbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/AssignAssignVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasabidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/Const*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
dtype0
Ô
cbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpReadVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ô
]bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/IdentityIdentitycbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ô
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
:
Ć
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *qÄž*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Ć
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
÷
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shape*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
š
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Í
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/sub*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ż
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ś
fbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelVarHandleOp*
shape:
*
dtype0*
_output_shapes
: *w
shared_namehfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
	container 

bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
ő
mbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/AssignAssignVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0

zbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpReadVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0* 
_output_shapes
:

§
tbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/IdentityIdentityzbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

Ö
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Č
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *qÄž*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Č
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
ú
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
seed2 
˝
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Ń
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/sub*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

Ă
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
š
gbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelVarHandleOp*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: *x
shared_nameigbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
 
bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpgbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
ů
nbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/AssignAssignVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform*
dtype0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Ą
{bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpReadVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0* 
_output_shapes
:

Š
ubidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/IdentityIdentity{bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

ž
vbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zerosConst*
valueB*    *w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:
Ť
dbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasVarHandleOp*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
	container *
shape:*
dtype0*
_output_shapes
: *u
shared_namefdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias

bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
_output_shapes
: 
ă
kbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/AssignAssignVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasvbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0

xbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpReadVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:

rbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/IdentityIdentityxbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ŕ
wbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias
Ž
ebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biasVarHandleOp*
dtype0*
_output_shapes
: *v
shared_namegebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
	container *
shape:

bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
_output_shapes
: 
ç
lbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/AssignAssignVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biaswbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0

ybidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOpReadVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
 
sbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/IdentityIdentityybidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ç
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat/axisConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concatConcatV2Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1(bidirectional_rnn/fw/fw/while/Identity_4\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat/axis*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
ň
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMulMatMulWbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat]bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/Enter*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ý
]bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/EnterEnter_bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ć
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAddBiasAddWbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/EnterEnter]bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
đ
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/SigmoidSigmoidXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/ConstConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ë
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split/split_dimConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ó
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/splitSplit`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split/split_dimXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Sigmoid*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split*
T0
ő
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1MatMulVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/Enter*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ô
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/EnterEntertbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ě
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1BiasAddYbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/EnterEnterrbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ç
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2MatMul(bidirectional_rnn/fw/fw/while/Identity_4_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/Enter*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ő
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/EnterEnterubidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ě
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2BiasAddYbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/EnterEntersbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
Â
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mulMulVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/splitZbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/addAddZbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ć
Ubidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/TanhTanhTbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub/xConst'^bidirectional_rnn/fw/fw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ŕ
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/subSubVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub/xXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_1MulTbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/subUbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_2MulXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:1(bidirectional_rnn/fw/fw/while/Identity_4*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1AddVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_1Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
rbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*
valueB"      *d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
:

pbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 

pbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 
ś
zbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniformrbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
seed2 
â
pbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/subSubpbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxpbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
ö
pbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulMulzbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformpbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/sub*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

č
lbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniformAddpbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulpbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

÷
Qbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelVarHandleOp*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: *b
shared_nameSQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel
ó
rbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
 
Xbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/AssignAssignVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernellbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0
ß
ebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpReadVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0* 
_output_shapes
:
*d
_classZ
XVloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel
ý
_bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/IdentityIdentityebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:


abidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/ConstConst*
dtype0*
_output_shapes	
:*
valueB*  ?*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias
ě
Obidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasVarHandleOp*`
shared_nameQObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
ď
pbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
_output_shapes
: 

Vbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/AssignAssignVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasabidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/Const*
dtype0*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias
Ô
cbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpReadVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ô
]bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/IdentityIdentitycbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ô
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Ć
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *qÄž*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Ć
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
÷
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shape*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
š
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
Í
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/sub*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ż
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ś
fbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelVarHandleOp*
dtype0*
_output_shapes
: *w
shared_namehfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
	container *
shape:


bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
ő
mbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/AssignAssignVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0

zbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpReadVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0* 
_output_shapes
:
*y
_classo
mkloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel
§
tbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/IdentityIdentityzbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

Ö
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
:
Č
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
Č
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *qÄ>*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
ú
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shape*

seed *
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
seed2 *
dtype0* 
_output_shapes
:

˝
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
Ń
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/sub*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

Ă
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

š
gbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelVarHandleOp*
dtype0*
_output_shapes
: *x
shared_nameigbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
	container *
shape:

 
bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpgbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
ů
nbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/AssignAssignVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0
Ą
{bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpReadVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0* 
_output_shapes
:

Š
ubidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/IdentityIdentity{bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOp* 
_output_shapes
:
*
T0
ž
vbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zerosConst*
valueB*    *w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:
Ť
dbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasVarHandleOp*
dtype0*
_output_shapes
: *u
shared_namefdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
	container *
shape:

bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
_output_shapes
: 
ă
kbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/AssignAssignVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasvbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0

xbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpReadVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:*w
_classm
kiloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias

rbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/IdentityIdentityxbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOp*
_output_shapes	
:*
T0
Ŕ
wbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zerosConst*
valueB*    *x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
Ž
ebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biasVarHandleOp*
dtype0*
_output_shapes
: *v
shared_namegebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
	container *
shape:

bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
_output_shapes
: 
ç
lbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/AssignAssignVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biaswbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros*
dtype0*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias

ybidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOpReadVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
 
sbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/IdentityIdentityybidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ç
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat/axisConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concatConcatV2Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1(bidirectional_rnn/fw/fw/while/Identity_5\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat/axis*
T0*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ň
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMulMatMulWbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat]bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/Enter*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
Ý
]bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/EnterEnter_bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ć
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAddBiasAddWbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/Enter*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
×
^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/EnterEnter]bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*
is_constant(
đ
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/SigmoidSigmoidXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/ConstConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ë
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split/split_dimConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ó
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/splitSplit`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split/split_dimXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Sigmoid*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split*
T0
ő
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1MatMulVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ô
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/EnterEntertbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ě
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1BiasAddYbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/EnterEnterrbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Ç
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2MatMul(bidirectional_rnn/fw/fw/while/Identity_5_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/Enter*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ő
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/EnterEnterubidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
ě
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2BiasAddYbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/EnterEntersbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context
Â
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mulMulVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/splitZbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/addAddZbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Ubidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/TanhTanhTbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub/xConst'^bidirectional_rnn/fw/fw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ŕ
Tbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/subSubVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub/xXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_1MulTbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/subUbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_2MulXbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:1(bidirectional_rnn/fw/fw/while/Identity_5*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1AddVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_1Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_2*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ü
$bidirectional_rnn/fw/fw/while/SelectSelect*bidirectional_rnn/fw/fw/while/GreaterEqual*bidirectional_rnn/fw/fw/while/Select/EnterVbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1
Ű
*bidirectional_rnn/fw/fw/while/Select/EnterEnterbidirectional_rnn/fw/fw/zeros*
parallel_iterations *
is_constant(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1
ü
&bidirectional_rnn/fw/fw/while/Select_1Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_3Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1
ü
&bidirectional_rnn/fw/fw/while/Select_2Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_4Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
&bidirectional_rnn/fw/fw/while/Select_3Select*bidirectional_rnn/fw/fw/while/GreaterEqual(bidirectional_rnn/fw/fw/while/Identity_5Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
Abidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Gbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter(bidirectional_rnn/fw/fw/while/Identity_1$bidirectional_rnn/fw/fw/while/Select(bidirectional_rnn/fw/fw/while/Identity_2*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*
_output_shapes
: 
đ
Gbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter#bidirectional_rnn/fw/fw/TensorArray*
parallel_iterations *
is_constant(*
_output_shapes
:*;

frame_name-+bidirectional_rnn/fw/fw/while/while_context*
T0*i
_class_
][loc:@bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1

%bidirectional_rnn/fw/fw/while/add_1/yConst'^bidirectional_rnn/fw/fw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

#bidirectional_rnn/fw/fw/while/add_1Add(bidirectional_rnn/fw/fw/while/Identity_1%bidirectional_rnn/fw/fw/while/add_1/y*
_output_shapes
: *
T0

+bidirectional_rnn/fw/fw/while/NextIterationNextIteration!bidirectional_rnn/fw/fw/while/add*
T0*
_output_shapes
: 

-bidirectional_rnn/fw/fw/while/NextIteration_1NextIteration#bidirectional_rnn/fw/fw/while/add_1*
T0*
_output_shapes
: 
˘
-bidirectional_rnn/fw/fw/while/NextIteration_2NextIterationAbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

-bidirectional_rnn/fw/fw/while/NextIteration_3NextIteration&bidirectional_rnn/fw/fw/while/Select_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-bidirectional_rnn/fw/fw/while/NextIteration_4NextIteration&bidirectional_rnn/fw/fw/while/Select_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-bidirectional_rnn/fw/fw/while/NextIteration_5NextIteration&bidirectional_rnn/fw/fw/while/Select_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
"bidirectional_rnn/fw/fw/while/ExitExit$bidirectional_rnn/fw/fw/while/Switch*
T0*
_output_shapes
: 
u
$bidirectional_rnn/fw/fw/while/Exit_1Exit&bidirectional_rnn/fw/fw/while/Switch_1*
_output_shapes
: *
T0
u
$bidirectional_rnn/fw/fw/while/Exit_2Exit&bidirectional_rnn/fw/fw/while/Switch_2*
_output_shapes
: *
T0

$bidirectional_rnn/fw/fw/while/Exit_3Exit&bidirectional_rnn/fw/fw/while/Switch_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

$bidirectional_rnn/fw/fw/while/Exit_4Exit&bidirectional_rnn/fw/fw/while/Switch_4*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

$bidirectional_rnn/fw/fw/while/Exit_5Exit&bidirectional_rnn/fw/fw/while/Switch_5*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
:bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3#bidirectional_rnn/fw/fw/TensorArray$bidirectional_rnn/fw/fw/while/Exit_2*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray
Ž
4bidirectional_rnn/fw/fw/TensorArrayStack/range/startConst*
value	B : *6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
Ž
4bidirectional_rnn/fw/fw/TensorArrayStack/range/deltaConst*
value	B :*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
dtype0*
_output_shapes
: 
Č
.bidirectional_rnn/fw/fw/TensorArrayStack/rangeRange4bidirectional_rnn/fw/fw/TensorArrayStack/range/start:bidirectional_rnn/fw/fw/TensorArrayStack/TensorArraySizeV34bidirectional_rnn/fw/fw/TensorArrayStack/range/delta*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ń
<bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3#bidirectional_rnn/fw/fw/TensorArray.bidirectional_rnn/fw/fw/TensorArrayStack/range$bidirectional_rnn/fw/fw/while/Exit_2*%
element_shape:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@bidirectional_rnn/fw/fw/TensorArray*
dtype0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
j
bidirectional_rnn/fw/fw/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
`
bidirectional_rnn/fw/fw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
%bidirectional_rnn/fw/fw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
g
%bidirectional_rnn/fw/fw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ž
bidirectional_rnn/fw/fw/range_1Range%bidirectional_rnn/fw/fw/range_1/startbidirectional_rnn/fw/fw/Rank_1%bidirectional_rnn/fw/fw/range_1/delta*

Tidx0*
_output_shapes
:
z
)bidirectional_rnn/fw/fw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
g
%bidirectional_rnn/fw/fw/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ů
 bidirectional_rnn/fw/fw/concat_2ConcatV2)bidirectional_rnn/fw/fw/concat_2/values_0bidirectional_rnn/fw/fw/range_1%bidirectional_rnn/fw/fw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ý
#bidirectional_rnn/fw/fw/transpose_1	Transpose<bidirectional_rnn/fw/fw/TensorArrayStack/TensorArrayGatherV3 bidirectional_rnn/fw/fw/concat_2*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tperm0
Ć
$bidirectional_rnn/bw/ReverseSequenceReverseSequenceembedding_lookup/Identitylengths*
	batch_dim *
T0*
seq_dim*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tlen0	
^
bidirectional_rnn/bw/bw/RankConst*
value	B :*
dtype0*
_output_shapes
: 
e
#bidirectional_rnn/bw/bw/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
e
#bidirectional_rnn/bw/bw/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ś
bidirectional_rnn/bw/bw/rangeRange#bidirectional_rnn/bw/bw/range/startbidirectional_rnn/bw/bw/Rank#bidirectional_rnn/bw/bw/range/delta*
_output_shapes
:*

Tidx0
x
'bidirectional_rnn/bw/bw/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
e
#bidirectional_rnn/bw/bw/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ń
bidirectional_rnn/bw/bw/concatConcatV2'bidirectional_rnn/bw/bw/concat/values_0bidirectional_rnn/bw/bw/range#bidirectional_rnn/bw/bw/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ŕ
!bidirectional_rnn/bw/bw/transpose	Transpose$bidirectional_rnn/bw/ReverseSequencebidirectional_rnn/bw/bw/concat*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tperm0
}
bidirectional_rnn/bw/bw/ToInt32Castlengths*

SrcT0	*
Truncate( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0

'bidirectional_rnn/bw/bw/sequence_lengthIdentitybidirectional_rnn/bw/bw/ToInt32*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
~
bidirectional_rnn/bw/bw/ShapeShape!bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
u
+bidirectional_rnn/bw/bw/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
w
-bidirectional_rnn/bw/bw/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
w
-bidirectional_rnn/bw/bw/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ń
%bidirectional_rnn/bw/bw/strided_sliceStridedSlicebidirectional_rnn/bw/bw/Shape+bidirectional_rnn/bw/bw/strided_slice/stack-bidirectional_rnn/bw/bw/strided_slice/stack_1-bidirectional_rnn/bw/bw/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
ś
tbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ě
pbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims
ExpandDims%bidirectional_rnn/bw/bw/strided_slicetbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
ś
kbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ł
qbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

lbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concatConcatV2pbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDimskbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/Constqbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
ś
qbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

kbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zerosFilllbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/concatqbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
¸
vbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Đ
rbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims_1
ExpandDims%bidirectional_rnn/bw/bw/strided_slicevbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dim*
T0*
_output_shapes
:*

Tdim0
¸
mbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
¸
vbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Đ
rbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims
ExpandDims%bidirectional_rnn/bw/bw/strided_slicevbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
¸
mbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ľ
sbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

nbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concatConcatV2rbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDimsmbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/Constsbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
¸
sbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

mbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zerosFillnbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/concatsbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
ş
xbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ô
tbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims_1
ExpandDims%bidirectional_rnn/bw/bw/strided_slicexbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
ş
obidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
¸
vbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Đ
rbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims
ExpandDims%bidirectional_rnn/bw/bw/strided_slicevbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims/dim*
T0*
_output_shapes
:*

Tdim0
¸
mbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:
ľ
sbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

nbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concatConcatV2rbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDimsmbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/Constsbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
¸
sbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

mbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zerosFillnbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/concatsbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
xbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B : 
Ô
tbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims_1
ExpandDims%bidirectional_rnn/bw/bw/strided_slicexbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
ş
obidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

bidirectional_rnn/bw/bw/Shape_1Shape'bidirectional_rnn/bw/bw/sequence_length*
T0*
out_type0*
_output_shapes
:

bidirectional_rnn/bw/bw/stackPack%bidirectional_rnn/bw/bw/strided_slice*
T0*

axis *
N*
_output_shapes
:

bidirectional_rnn/bw/bw/EqualEqualbidirectional_rnn/bw/bw/Shape_1bidirectional_rnn/bw/bw/stack*
T0*
_output_shapes
:
g
bidirectional_rnn/bw/bw/ConstConst*
valueB: *
dtype0*
_output_shapes
:

bidirectional_rnn/bw/bw/AllAllbidirectional_rnn/bw/bw/Equalbidirectional_rnn/bw/bw/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
Ź
$bidirectional_rnn/bw/bw/Assert/ConstConst*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 
w
&bidirectional_rnn/bw/bw/Assert/Const_1Const*
dtype0*
_output_shapes
: *!
valueB B but saw shape: 
´
,bidirectional_rnn/bw/bw/Assert/Assert/data_0Const*X
valueOBM BGExpected shape for Tensor bidirectional_rnn/bw/bw/sequence_length:0 is *
dtype0*
_output_shapes
: 
}
,bidirectional_rnn/bw/bw/Assert/Assert/data_2Const*!
valueB B but saw shape: *
dtype0*
_output_shapes
: 

%bidirectional_rnn/bw/bw/Assert/AssertAssertbidirectional_rnn/bw/bw/All,bidirectional_rnn/bw/bw/Assert/Assert/data_0bidirectional_rnn/bw/bw/stack,bidirectional_rnn/bw/bw/Assert/Assert/data_2bidirectional_rnn/bw/bw/Shape_1*
T
2*
	summarize
Ž
#bidirectional_rnn/bw/bw/CheckSeqLenIdentity'bidirectional_rnn/bw/bw/sequence_length&^bidirectional_rnn/bw/bw/Assert/Assert*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

bidirectional_rnn/bw/bw/Shape_2Shape!bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
w
-bidirectional_rnn/bw/bw/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
y
/bidirectional_rnn/bw/bw/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/bidirectional_rnn/bw/bw/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ű
'bidirectional_rnn/bw/bw/strided_slice_1StridedSlicebidirectional_rnn/bw/bw/Shape_2-bidirectional_rnn/bw/bw/strided_slice_1/stack/bidirectional_rnn/bw/bw/strided_slice_1/stack_1/bidirectional_rnn/bw/bw/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

bidirectional_rnn/bw/bw/Shape_3Shape!bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:
w
-bidirectional_rnn/bw/bw/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
y
/bidirectional_rnn/bw/bw/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
y
/bidirectional_rnn/bw/bw/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ű
'bidirectional_rnn/bw/bw/strided_slice_2StridedSlicebidirectional_rnn/bw/bw/Shape_3-bidirectional_rnn/bw/bw/strided_slice_2/stack/bidirectional_rnn/bw/bw/strided_slice_2/stack_1/bidirectional_rnn/bw/bw/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
h
&bidirectional_rnn/bw/bw/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
˛
"bidirectional_rnn/bw/bw/ExpandDims
ExpandDims'bidirectional_rnn/bw/bw/strided_slice_2&bidirectional_rnn/bw/bw/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
j
bidirectional_rnn/bw/bw/Const_1Const*
dtype0*
_output_shapes
:*
valueB:
g
%bidirectional_rnn/bw/bw/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ň
 bidirectional_rnn/bw/bw/concat_1ConcatV2"bidirectional_rnn/bw/bw/ExpandDimsbidirectional_rnn/bw/bw/Const_1%bidirectional_rnn/bw/bw/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
h
#bidirectional_rnn/bw/bw/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ą
bidirectional_rnn/bw/bw/zerosFill bidirectional_rnn/bw/bw/concat_1#bidirectional_rnn/bw/bw/zeros/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
i
bidirectional_rnn/bw/bw/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Ś
bidirectional_rnn/bw/bw/MinMin#bidirectional_rnn/bw/bw/CheckSeqLenbidirectional_rnn/bw/bw/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
bidirectional_rnn/bw/bw/Const_3Const*
valueB: *
dtype0*
_output_shapes
:
Ś
bidirectional_rnn/bw/bw/MaxMax#bidirectional_rnn/bw/bw/CheckSeqLenbidirectional_rnn/bw/bw/Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
^
bidirectional_rnn/bw/bw/timeConst*
dtype0*
_output_shapes
: *
value	B : 
Ŕ
#bidirectional_rnn/bw/bw/TensorArrayTensorArrayV3'bidirectional_rnn/bw/bw/strided_slice_1*C
tensor_array_name.,bidirectional_rnn/bw/bw/dynamic_rnn/output_0*
dtype0*
_output_shapes

:: *%
element_shape:˙˙˙˙˙˙˙˙˙*
clear_after_read(*
dynamic_size( *
identical_element_shapes(
Ŕ
%bidirectional_rnn/bw/bw/TensorArray_1TensorArrayV3'bidirectional_rnn/bw/bw/strided_slice_1*B
tensor_array_name-+bidirectional_rnn/bw/bw/dynamic_rnn/input_0*
dtype0*
_output_shapes

:: *$
element_shape:˙˙˙˙˙˙˙˙˙*
dynamic_size( *
clear_after_read(*
identical_element_shapes(

0bidirectional_rnn/bw/bw/TensorArrayUnstack/ShapeShape!bidirectional_rnn/bw/bw/transpose*
T0*
out_type0*
_output_shapes
:

>bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Đ
8bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_sliceStridedSlice0bidirectional_rnn/bw/bw/TensorArrayUnstack/Shape>bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_1@bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
x
6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 

0bidirectional_rnn/bw/bw/TensorArrayUnstack/rangeRange6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/start8bidirectional_rnn/bw/bw/TensorArrayUnstack/strided_slice6bidirectional_rnn/bw/bw/TensorArrayUnstack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ć
Rbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3%bidirectional_rnn/bw/bw/TensorArray_10bidirectional_rnn/bw/bw/TensorArrayUnstack/range!bidirectional_rnn/bw/bw/transpose'bidirectional_rnn/bw/bw/TensorArray_1:1*
_output_shapes
: *
T0*4
_class*
(&loc:@bidirectional_rnn/bw/bw/transpose
c
!bidirectional_rnn/bw/bw/Maximum/xConst*
dtype0*
_output_shapes
: *
value	B :

bidirectional_rnn/bw/bw/MaximumMaximum!bidirectional_rnn/bw/bw/Maximum/xbidirectional_rnn/bw/bw/Max*
T0*
_output_shapes
: 

bidirectional_rnn/bw/bw/MinimumMinimum'bidirectional_rnn/bw/bw/strided_slice_1bidirectional_rnn/bw/bw/Maximum*
T0*
_output_shapes
: 
q
/bidirectional_rnn/bw/bw/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
é
#bidirectional_rnn/bw/bw/while/EnterEnter/bidirectional_rnn/bw/bw/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ř
%bidirectional_rnn/bw/bw/while/Enter_1Enterbidirectional_rnn/bw/bw/time*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
á
%bidirectional_rnn/bw/bw/while/Enter_2Enter%bidirectional_rnn/bw/bw/TensorArray:1*
T0*
is_constant( *
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
š
%bidirectional_rnn/bw/bw/while/Enter_3Enterkbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState/CudnnCompatibleGRUCellZeroState/zeros*
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant( 
ť
%bidirectional_rnn/bw/bw/while/Enter_4Entermbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_1/CudnnCompatibleGRUCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ť
%bidirectional_rnn/bw/bw/while/Enter_5Entermbidirectional_rnn/bw/bw/MultiRNNCellZeroState/DropoutWrapperZeroState_2/CudnnCompatibleGRUCellZeroState/zeros*
T0*
is_constant( *
parallel_iterations *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ş
#bidirectional_rnn/bw/bw/while/MergeMerge#bidirectional_rnn/bw/bw/while/Enter+bidirectional_rnn/bw/bw/while/NextIteration*
N*
_output_shapes
: : *
T0
°
%bidirectional_rnn/bw/bw/while/Merge_1Merge%bidirectional_rnn/bw/bw/while/Enter_1-bidirectional_rnn/bw/bw/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
°
%bidirectional_rnn/bw/bw/while/Merge_2Merge%bidirectional_rnn/bw/bw/while/Enter_2-bidirectional_rnn/bw/bw/while/NextIteration_2*
N*
_output_shapes
: : *
T0
Â
%bidirectional_rnn/bw/bw/while/Merge_3Merge%bidirectional_rnn/bw/bw/while/Enter_3-bidirectional_rnn/bw/bw/while/NextIteration_3*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Â
%bidirectional_rnn/bw/bw/while/Merge_4Merge%bidirectional_rnn/bw/bw/while/Enter_4-bidirectional_rnn/bw/bw/while/NextIteration_4*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 
Â
%bidirectional_rnn/bw/bw/while/Merge_5Merge%bidirectional_rnn/bw/bw/while/Enter_5-bidirectional_rnn/bw/bw/while/NextIteration_5*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

"bidirectional_rnn/bw/bw/while/LessLess#bidirectional_rnn/bw/bw/while/Merge(bidirectional_rnn/bw/bw/while/Less/Enter*
_output_shapes
: *
T0
ć
(bidirectional_rnn/bw/bw/while/Less/EnterEnter'bidirectional_rnn/bw/bw/strided_slice_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
 
$bidirectional_rnn/bw/bw/while/Less_1Less%bidirectional_rnn/bw/bw/while/Merge_1*bidirectional_rnn/bw/bw/while/Less_1/Enter*
T0*
_output_shapes
: 
ŕ
*bidirectional_rnn/bw/bw/while/Less_1/EnterEnterbidirectional_rnn/bw/bw/Minimum*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context

(bidirectional_rnn/bw/bw/while/LogicalAnd
LogicalAnd"bidirectional_rnn/bw/bw/while/Less$bidirectional_rnn/bw/bw/while/Less_1*
_output_shapes
: 
t
&bidirectional_rnn/bw/bw/while/LoopCondLoopCond(bidirectional_rnn/bw/bw/while/LogicalAnd*
_output_shapes
: 
Ö
$bidirectional_rnn/bw/bw/while/SwitchSwitch#bidirectional_rnn/bw/bw/while/Merge&bidirectional_rnn/bw/bw/while/LoopCond*
T0*6
_class,
*(loc:@bidirectional_rnn/bw/bw/while/Merge*
_output_shapes
: : 
Ü
&bidirectional_rnn/bw/bw/while/Switch_1Switch%bidirectional_rnn/bw/bw/while/Merge_1&bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_1
Ü
&bidirectional_rnn/bw/bw/while/Switch_2Switch%bidirectional_rnn/bw/bw/while/Merge_2&bidirectional_rnn/bw/bw/while/LoopCond*
_output_shapes
: : *
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_2

&bidirectional_rnn/bw/bw/while/Switch_3Switch%bidirectional_rnn/bw/bw/while/Merge_3&bidirectional_rnn/bw/bw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_3*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

&bidirectional_rnn/bw/bw/while/Switch_4Switch%bidirectional_rnn/bw/bw/while/Merge_4&bidirectional_rnn/bw/bw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_4*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

&bidirectional_rnn/bw/bw/while/Switch_5Switch%bidirectional_rnn/bw/bw/while/Merge_5&bidirectional_rnn/bw/bw/while/LoopCond*
T0*8
_class.
,*loc:@bidirectional_rnn/bw/bw/while/Merge_5*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
{
&bidirectional_rnn/bw/bw/while/IdentityIdentity&bidirectional_rnn/bw/bw/while/Switch:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_1Identity(bidirectional_rnn/bw/bw/while/Switch_1:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_2Identity(bidirectional_rnn/bw/bw/while/Switch_2:1*
T0*
_output_shapes
: 

(bidirectional_rnn/bw/bw/while/Identity_3Identity(bidirectional_rnn/bw/bw/while/Switch_3:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

(bidirectional_rnn/bw/bw/while/Identity_4Identity(bidirectional_rnn/bw/bw/while/Switch_4:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

(bidirectional_rnn/bw/bw/while/Identity_5Identity(bidirectional_rnn/bw/bw/while/Switch_5:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

#bidirectional_rnn/bw/bw/while/add/yConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

!bidirectional_rnn/bw/bw/while/addAdd&bidirectional_rnn/bw/bw/while/Identity#bidirectional_rnn/bw/bw/while/add/y*
T0*
_output_shapes
: 

/bidirectional_rnn/bw/bw/while/TensorArrayReadV3TensorArrayReadV35bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter(bidirectional_rnn/bw/bw/while/Identity_17bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ő
5bidirectional_rnn/bw/bw/while/TensorArrayReadV3/EnterEnter%bidirectional_rnn/bw/bw/TensorArray_1*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
 
7bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1EnterRbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *
_output_shapes
: *;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ä
*bidirectional_rnn/bw/bw/while/GreaterEqualGreaterEqual(bidirectional_rnn/bw/bw/while/Identity_10bidirectional_rnn/bw/bw/while/GreaterEqual/Enter*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
÷
0bidirectional_rnn/bw/bw/while/GreaterEqual/EnterEnter#bidirectional_rnn/bw/bw/CheckSeqLen*
T0*
is_constant(*
parallel_iterations *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Š
rbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel

pbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/minConst*
valueB
 *Ń_ý˝*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 

pbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ń_ý=*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 
ś
zbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniformrbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shape*

seed *
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
seed2 *
dtype0* 
_output_shapes
:

â
pbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/subSubpbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxpbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
ö
pbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulMulzbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformpbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/sub*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

č
lbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniformAddpbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulpbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

÷
Qbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelVarHandleOp*b
shared_nameSQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
	container *
shape:
*
dtype0*
_output_shapes
: 
ó
rbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
 
Xbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/AssignAssignVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernellbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform*
dtype0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel
ß
ebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpReadVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel*
dtype0* 
_output_shapes
:

ý
_bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/IdentityIdentityebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:


abidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/ConstConst*
dtype0*
_output_shapes	
:*
valueB*  ?*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias
ě
Obidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasVarHandleOp*`
shared_nameQObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
ď
pbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
_output_shapes
: 

Vbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/AssignAssignVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasabidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/Const*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
dtype0
Ô
cbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpReadVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ô
]bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/IdentityIdentitycbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ô
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
:
Ć
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *AWž*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
Ć
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *AW>*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
ö
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
seed2 
š
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel
Ě
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/sub*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
:	
ž
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel
ľ
fbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelVarHandleOp*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
	container *
shape:	*
dtype0*
_output_shapes
: *w
shared_namehfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel

bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
ő
mbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/AssignAssignVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0

zbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpReadVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
:	
Ś
tbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/IdentityIdentityzbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOp*
T0*
_output_shapes
:	
Ö
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Č
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
Č
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
ú
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shape*

seed *
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
seed2 *
dtype0* 
_output_shapes
:

˝
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
Ń
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/sub*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

Ă
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

š
gbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelVarHandleOp*
dtype0*
_output_shapes
: *x
shared_nameigbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
	container *
shape:

 
bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpgbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
ů
nbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/AssignAssignVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0
Ą
{bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpReadVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0* 
_output_shapes
:

Š
ubidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/IdentityIdentity{bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

ž
vbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zerosConst*
valueB*    *w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:
Ť
dbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasVarHandleOp*u
shared_namefdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
	container *
shape:*
dtype0*
_output_shapes
: 

bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
_output_shapes
: 
ă
kbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/AssignAssignVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasvbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0

xbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpReadVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:

rbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/IdentityIdentityxbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOp*
_output_shapes	
:*
T0
Ŕ
wbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zerosConst*
valueB*    *x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
Ž
ebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *v
shared_namegebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
	container 

bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
_output_shapes
: 
ç
lbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/AssignAssignVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biaswbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0

ybidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOpReadVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
 
sbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/IdentityIdentityybidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ç
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat/axisConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ä
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concatConcatV2/bidirectional_rnn/bw/bw/while/TensorArrayReadV3(bidirectional_rnn/bw/bw/while/Identity_3\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat/axis*
T0*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ň
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMulMatMulWbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat]bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/Enter*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
Ý
]bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/EnterEnter_bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ć
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAddBiasAddWbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/EnterEnter]bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
đ
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/SigmoidSigmoidXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/ConstConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Ë
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split/split_dimConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ó
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/splitSplit`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split/split_dimXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Sigmoid*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split
Î
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1MatMul/bidirectional_rnn/bw/bw/while/TensorArrayReadV3_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ó
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/EnterEntertbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes
:	*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ě
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1BiasAddYbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/EnterEnterrbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ç
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2MatMul(bidirectional_rnn/bw/bw/while/Identity_3_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/Enter*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
ő
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/EnterEnterubidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ě
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2BiasAddYbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/Enter*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ď
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/EnterEntersbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Â
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mulMulVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/splitZbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/addAddZbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Ubidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/TanhTanhTbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub/xConst'^bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ŕ
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/subSubVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub/xXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_1MulTbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/subUbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_2MulXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:1(bidirectional_rnn/bw/bw/while/Identity_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1AddVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_1Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
rbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*
valueB"      *d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
:

pbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *×łÝ˝*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel

pbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxConst*
valueB
 *×łÝ=*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 
ś
zbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniformrbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
seed2 
â
pbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/subSubpbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxpbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
ö
pbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulMulzbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformpbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/sub*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

č
lbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniformAddpbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulpbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

÷
Qbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelVarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *b
shared_nameSQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel
ó
rbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
 
Xbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/AssignAssignVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernellbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0
ß
ebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpReadVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel*
dtype0* 
_output_shapes
:

ý
_bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/IdentityIdentityebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:


abidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/ConstConst*
dtype0*
_output_shapes	
:*
valueB*  ?*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias
ě
Obidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasVarHandleOp*`
shared_nameQObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
ď
pbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
_output_shapes
: 

Vbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/AssignAssignVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasabidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/Const*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
dtype0
Ô
cbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpReadVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ô
]bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/IdentityIdentitycbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ô
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
:
Ć
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
Ć
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *qÄ>*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel
÷
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
seed2 
š
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
Í
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/sub*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ż
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ś
fbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelVarHandleOp*
dtype0*
_output_shapes
: *w
shared_namehfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
	container *
shape:


bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
ő
mbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/AssignAssignVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0

zbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpReadVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0* 
_output_shapes
:

§
tbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/IdentityIdentityzbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOp* 
_output_shapes
:
*
T0
Ö
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Č
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
Č
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
ú
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
seed2 
˝
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
Ń
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/sub*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

Ă
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

š
gbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelVarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *x
shared_nameigbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
 
bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpgbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
ů
nbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/AssignAssignVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0
Ą
{bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpReadVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0* 
_output_shapes
:

Š
ubidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/IdentityIdentity{bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

ž
vbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zerosConst*
valueB*    *w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:
Ť
dbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasVarHandleOp*
dtype0*
_output_shapes
: *u
shared_namefdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
	container *
shape:

bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
_output_shapes
: 
ă
kbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/AssignAssignVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasvbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros*
dtype0*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias

xbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpReadVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:

rbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/IdentityIdentityxbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ŕ
wbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias
Ž
ebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biasVarHandleOp*
	container *
shape:*
dtype0*
_output_shapes
: *v
shared_namegebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias

bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
_output_shapes
: 
ç
lbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/AssignAssignVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biaswbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros*
dtype0*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias

ybidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOpReadVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
 
sbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/IdentityIdentityybidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ç
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat/axisConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concatConcatV2Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1(bidirectional_rnn/bw/bw/while/Identity_4\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat/axis*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
ň
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMulMatMulWbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat]bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/Enter*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0
Ý
]bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/EnterEnter_bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ć
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAddBiasAddWbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/EnterEnter]bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
đ
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/SigmoidSigmoidXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/ConstConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ë
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split/split_dimConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ó
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/splitSplit`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split/split_dimXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Sigmoid*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split
ő
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1MatMulVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ô
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/EnterEntertbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
ě
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1BiasAddYbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/EnterEnterrbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ç
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2MatMul(bidirectional_rnn/bw/bw/while/Identity_4_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ő
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/EnterEnterubidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ě
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2BiasAddYbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/EnterEntersbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*
is_constant(
Â
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mulMulVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/splitZbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/addAddZbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Ubidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/TanhTanhTbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub/xConst'^bidirectional_rnn/bw/bw/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ŕ
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/subSubVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub/xXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_1MulTbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/subUbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_2MulXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:1(bidirectional_rnn/bw/bw/while/Identity_4*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1AddVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_1Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
rbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shapeConst*
valueB"      *d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
:

pbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/minConst*
valueB
 *×łÝ˝*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0*
_output_shapes
: 

pbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *×łÝ=*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel
ś
zbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformRandomUniformrbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
seed2 
â
pbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/subSubpbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/maxpbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
ö
pbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulMulzbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/RandomUniformpbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/sub*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

č
lbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniformAddpbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/mulpbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform/min*
T0*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel* 
_output_shapes
:

÷
Qbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelVarHandleOp*
dtype0*
_output_shapes
: *b
shared_nameSQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
	container *
shape:

ó
rbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
_output_shapes
: 
 
Xbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/AssignAssignVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernellbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0
ß
ebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpReadVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*d
_classZ
XVloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel*
dtype0* 
_output_shapes
:

ý
_bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/IdentityIdentityebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:


abidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/ConstConst*
valueB*  ?*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:
ě
Obidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *`
shared_nameQObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
	container 
ď
pbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
_output_shapes
: 

Vbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/AssignAssignVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasabidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/Const*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
dtype0
Ô
cbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpReadVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias*
dtype0*
_output_shapes	
:*b
_classX
VTloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias
ô
]bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/IdentityIdentitycbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ô
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
:
Ć
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
Ć
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0*
_output_shapes
: 
÷
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
seed2 
š
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
Í
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/sub*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel* 
_output_shapes
:

ż
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel
ś
fbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelVarHandleOp*
dtype0*
_output_shapes
: *w
shared_namehfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
	container *
shape:


bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
_output_shapes
: 
ő
mbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/AssignAssignVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0

zbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpReadVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel*
dtype0* 
_output_shapes
:
*y
_classo
mkloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel
§
tbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/IdentityIdentityzbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

Ö
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shapeConst*
valueB"      *z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
:
Č
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
Č
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0*
_output_shapes
: 
ú
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformRandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
seed2 
˝
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/subSubbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/maxbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
Ń
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulMulbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/RandomUniformbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/sub*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel* 
_output_shapes
:

Ă
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniformAddbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/mulbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
š
gbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelVarHandleOp*
	container *
shape:
*
dtype0*
_output_shapes
: *x
shared_nameigbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel
 
bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpgbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
_output_shapes
: 
ů
nbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/AssignAssignVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0
Ą
{bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpReadVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*z
_classp
nlloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel*
dtype0* 
_output_shapes
:

Š
ubidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/IdentityIdentity{bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOp*
T0* 
_output_shapes
:

ž
vbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zerosConst*
valueB*    *w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:
Ť
dbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *u
shared_namefdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
	container 

bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
_output_shapes
: 
ă
kbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/AssignAssignVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasvbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0

xbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpReadVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*w
_classm
kiloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias*
dtype0*
_output_shapes	
:

rbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/IdentityIdentityxbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ŕ
wbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zerosConst*
valueB*    *x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
Ž
ebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biasVarHandleOp*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
	container *
shape:*
dtype0*
_output_shapes
: *v
shared_namegebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias

bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
_output_shapes
: 
ç
lbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/AssignAssignVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biaswbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0

ybidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOpReadVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*x
_classn
ljloc:@bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias*
dtype0*
_output_shapes	
:
 
sbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/IdentityIdentityybidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp*
T0*
_output_shapes	
:
Ç
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat/axisConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concatConcatV2Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1(bidirectional_rnn/bw/bw/while/Identity_5\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat/axis*

Tidx0*
T0*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ň
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMulMatMulWbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat]bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
Ý
]bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/EnterEnter_bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ć
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAddBiasAddWbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
×
^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/EnterEnter]bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
đ
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/SigmoidSigmoidXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Á
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/ConstConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Ë
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split/split_dimConst'^bidirectional_rnn/bw/bw/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
ó
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/splitSplit`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split/split_dimXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Sigmoid*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
	num_split
ő
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1MatMulVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/Enter*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 
ô
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/EnterEntertbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ě
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1BiasAddYbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
î
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/EnterEnterrbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Ç
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2MatMul(bidirectional_rnn/bw/bw/while/Identity_5_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/Enter*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ő
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/EnterEnterubidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity*
T0*
is_constant(*
parallel_iterations * 
_output_shapes
:
*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
ě
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2BiasAddYbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/Enter*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/EnterEntersbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity*
T0*
is_constant(*
parallel_iterations *
_output_shapes	
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context
Â
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mulMulVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/splitZbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/addAddZbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Ubidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/TanhTanhTbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub/xConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *  ?
Ŕ
Tbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/subSubVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub/xXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_1MulTbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/subUbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Tanh*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_2MulXbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:1(bidirectional_rnn/bw/bw/while/Identity_5*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1AddVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_1Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
$bidirectional_rnn/bw/bw/while/SelectSelect*bidirectional_rnn/bw/bw/while/GreaterEqual*bidirectional_rnn/bw/bw/while/Select/EnterVbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
*bidirectional_rnn/bw/bw/while/Select/EnterEnterbidirectional_rnn/bw/bw/zeros*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*
parallel_iterations *
is_constant(
ü
&bidirectional_rnn/bw/bw/while/Select_1Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_3Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1*
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
&bidirectional_rnn/bw/bw/while/Select_2Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_4Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1*
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
&bidirectional_rnn/bw/bw/while/Select_3Select*bidirectional_rnn/bw/bw/while/GreaterEqual(bidirectional_rnn/bw/bw/while/Identity_5Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1
Ś
Abidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Gbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter(bidirectional_rnn/bw/bw/while/Identity_1$bidirectional_rnn/bw/bw/while/Select(bidirectional_rnn/bw/bw/while/Identity_2*
_output_shapes
: *
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1
đ
Gbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter#bidirectional_rnn/bw/bw/TensorArray*
_output_shapes
:*;

frame_name-+bidirectional_rnn/bw/bw/while/while_context*
T0*i
_class_
][loc:@bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1*
parallel_iterations *
is_constant(

%bidirectional_rnn/bw/bw/while/add_1/yConst'^bidirectional_rnn/bw/bw/while/Identity*
dtype0*
_output_shapes
: *
value	B :

#bidirectional_rnn/bw/bw/while/add_1Add(bidirectional_rnn/bw/bw/while/Identity_1%bidirectional_rnn/bw/bw/while/add_1/y*
T0*
_output_shapes
: 

+bidirectional_rnn/bw/bw/while/NextIterationNextIteration!bidirectional_rnn/bw/bw/while/add*
_output_shapes
: *
T0

-bidirectional_rnn/bw/bw/while/NextIteration_1NextIteration#bidirectional_rnn/bw/bw/while/add_1*
T0*
_output_shapes
: 
˘
-bidirectional_rnn/bw/bw/while/NextIteration_2NextIterationAbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

-bidirectional_rnn/bw/bw/while/NextIteration_3NextIteration&bidirectional_rnn/bw/bw/while/Select_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

-bidirectional_rnn/bw/bw/while/NextIteration_4NextIteration&bidirectional_rnn/bw/bw/while/Select_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-bidirectional_rnn/bw/bw/while/NextIteration_5NextIteration&bidirectional_rnn/bw/bw/while/Select_3*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
q
"bidirectional_rnn/bw/bw/while/ExitExit$bidirectional_rnn/bw/bw/while/Switch*
T0*
_output_shapes
: 
u
$bidirectional_rnn/bw/bw/while/Exit_1Exit&bidirectional_rnn/bw/bw/while/Switch_1*
T0*
_output_shapes
: 
u
$bidirectional_rnn/bw/bw/while/Exit_2Exit&bidirectional_rnn/bw/bw/while/Switch_2*
T0*
_output_shapes
: 

$bidirectional_rnn/bw/bw/while/Exit_3Exit&bidirectional_rnn/bw/bw/while/Switch_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

$bidirectional_rnn/bw/bw/while/Exit_4Exit&bidirectional_rnn/bw/bw/while/Switch_4*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

$bidirectional_rnn/bw/bw/while/Exit_5Exit&bidirectional_rnn/bw/bw/while/Switch_5*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
:bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3#bidirectional_rnn/bw/bw/TensorArray$bidirectional_rnn/bw/bw/while/Exit_2*
_output_shapes
: *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
Ž
4bidirectional_rnn/bw/bw/TensorArrayStack/range/startConst*
dtype0*
_output_shapes
: *
value	B : *6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
Ž
4bidirectional_rnn/bw/bw/TensorArrayStack/range/deltaConst*
value	B :*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray*
dtype0*
_output_shapes
: 
Č
.bidirectional_rnn/bw/bw/TensorArrayStack/rangeRange4bidirectional_rnn/bw/bw/TensorArrayStack/range/start:bidirectional_rnn/bw/bw/TensorArrayStack/TensorArraySizeV34bidirectional_rnn/bw/bw/TensorArrayStack/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
ń
<bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3#bidirectional_rnn/bw/bw/TensorArray.bidirectional_rnn/bw/bw/TensorArrayStack/range$bidirectional_rnn/bw/bw/while/Exit_2*
dtype0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
element_shape:˙˙˙˙˙˙˙˙˙*6
_class,
*(loc:@bidirectional_rnn/bw/bw/TensorArray
j
bidirectional_rnn/bw/bw/Const_4Const*
valueB:*
dtype0*
_output_shapes
:
`
bidirectional_rnn/bw/bw/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
g
%bidirectional_rnn/bw/bw/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
g
%bidirectional_rnn/bw/bw/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
ž
bidirectional_rnn/bw/bw/range_1Range%bidirectional_rnn/bw/bw/range_1/startbidirectional_rnn/bw/bw/Rank_1%bidirectional_rnn/bw/bw/range_1/delta*
_output_shapes
:*

Tidx0
z
)bidirectional_rnn/bw/bw/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
g
%bidirectional_rnn/bw/bw/concat_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ů
 bidirectional_rnn/bw/bw/concat_2ConcatV2)bidirectional_rnn/bw/bw/concat_2/values_0bidirectional_rnn/bw/bw/range_1%bidirectional_rnn/bw/bw/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ý
#bidirectional_rnn/bw/bw/transpose_1	Transpose<bidirectional_rnn/bw/bw/TensorArrayStack/TensorArrayGatherV3 bidirectional_rnn/bw/bw/concat_2*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tperm0
ź
ReverseSequenceReverseSequence#bidirectional_rnn/bw/bw/transpose_1lengths*
	batch_dim *
T0*
seq_dim*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tlen0	
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
Ş
concatConcatV2#bidirectional_rnn/fw/fw/transpose_1ReverseSequenceconcat/axis*
T0*
N*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tidx0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *žož*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *žo>*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*
_class
loc:@dense/kernel
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
Ł
dense/kernel
VariableV2*
shape:	*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@dense/kernel*
	container 
Č
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:


dense/bias
VariableV2*
shared_name *
_class
loc:@dense/bias*
	container *
shape:*
dtype0*
_output_shapes
:
˛
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:
k
dense/bias/readIdentity
dense/bias*
_output_shapes
:*
T0*
_class
loc:@dense/bias
^
dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
e
dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
[
dense/Tensordot/ShapeShapeconcat*
T0*
out_type0*
_output_shapes
:
_
dense/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
¸
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shapedense/Tensordot/freedense/Tensordot/GatherV2/axis*
Tindices0*
Tparams0*
_output_shapes
:*
Taxis0
a
dense/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
ź
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shapedense/Tensordot/axesdense/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
_
dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:

dense/Tensordot/ProdProddense/Tensordot/GatherV2dense/Tensordot/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
a
dense/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

dense/Tensordot/Prod_1Proddense/Tensordot/GatherV2_1dense/Tensordot/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
]
dense/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
Ľ
dense/Tensordot/concatConcatV2dense/Tensordot/freedense/Tensordot/axesdense/Tensordot/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

dense/Tensordot/stackPackdense/Tensordot/Proddense/Tensordot/Prod_1*
T0*

axis *
N*
_output_shapes
:

dense/Tensordot/transpose	Transposeconcatdense/Tensordot/concat*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
Tperm0

dense/Tensordot/ReshapeReshapedense/Tensordot/transposedense/Tensordot/stack*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
q
 dense/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       

dense/Tensordot/transpose_1	Transposedense/kernel/read dense/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes
:	
p
dense/Tensordot/Reshape_1/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1dense/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes
:	
Ź
dense/Tensordot/MatMulMatMuldense/Tensordot/Reshapedense/Tensordot/Reshape_1*
transpose_b( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
a
dense/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
_
dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
°
dense/Tensordot/concat_1ConcatV2dense/Tensordot/GatherV2dense/Tensordot/Const_2dense/Tensordot/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0

dense/TensordotReshapedense/Tensordot/MatMuldense/Tensordot/concat_1*
T0*
Tshape0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

dense/BiasAddBiasAdddense/Tensordotdense/bias/read*
T0*
data_formatNHWC*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
^
	dense/EluEludense/BiasAdd*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMaxArgMax	dense/EluArgMax/dimension*
T0*
output_type0	*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

Tidx0
\
SoftmaxSoftmax	dense/Elu*
T0*4
_output_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_89aa9cddb3154728b802d3b91fc7babf/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
°
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:(*Ô
valueĘBÇ(Bebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelB
dense/biasBdense/kernelB
embeddingsBglobal_step
Â
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
š"
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesybidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp{bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpxbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpzbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpcbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpybidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp{bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpxbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpzbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpcbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpybidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp{bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpxbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpzbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpcbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpybidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp{bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpxbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpzbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpcbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpybidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp{bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpxbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpzbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpcbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOpybidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/ReadVariableOp{bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/ReadVariableOpxbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/ReadVariableOpzbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/ReadVariableOpcbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/ReadVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/ReadVariableOp
dense/biasdense/kernel
embeddingsglobal_step"/device:CPU:0*6
dtypes,
*2(	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
ł
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ô
valueĘBÇ(Bebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelBebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biasBgbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelBdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biasBfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelBObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biasBQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelB
dense/biasBdense/kernelB
embeddingsBglobal_step*
dtype0*
_output_shapes
:(
Ĺ
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
ĺ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*ś
_output_shapesŁ
 ::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
Ž
save/AssignVariableOpAssignVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
˛
save/AssignVariableOp_1AssignVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
Ż
save/AssignVariableOp_2AssignVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
ą
save/AssignVariableOp_3AssignVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:

save/AssignVariableOp_4AssignVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:

save/AssignVariableOp_5AssignVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
°
save/AssignVariableOp_6AssignVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
˛
save/AssignVariableOp_7AssignVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
_output_shapes
:*
T0
Ż
save/AssignVariableOp_8AssignVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biassave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
˛
save/AssignVariableOp_9AssignVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:

save/AssignVariableOp_10AssignVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:

save/AssignVariableOp_11AssignVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
_output_shapes
:*
T0
˛
save/AssignVariableOp_12AssignVariableOpebidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biassave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
´
save/AssignVariableOp_13AssignVariableOpgbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelsave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
ą
save/AssignVariableOp_14AssignVariableOpdbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biassave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
ł
save/AssignVariableOp_15AssignVariableOpfbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelsave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:

save/AssignVariableOp_16AssignVariableOpObidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biassave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:

save/AssignVariableOp_17AssignVariableOpQbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
˛
save/AssignVariableOp_18AssignVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/biassave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
_output_shapes
:*
T0
´
save/AssignVariableOp_19AssignVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelsave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
ą
save/AssignVariableOp_20AssignVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/biassave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
ł
save/AssignVariableOp_21AssignVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernelsave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:

save/AssignVariableOp_22AssignVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/biassave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:

save/AssignVariableOp_23AssignVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernelsave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
_output_shapes
:*
T0
˛
save/AssignVariableOp_24AssignVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/biassave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
_output_shapes
:*
T0
´
save/AssignVariableOp_25AssignVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelsave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
ą
save/AssignVariableOp_26AssignVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/biassave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
ł
save/AssignVariableOp_27AssignVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernelsave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:

save/AssignVariableOp_28AssignVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/biassave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:

save/AssignVariableOp_29AssignVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernelsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
˛
save/AssignVariableOp_30AssignVariableOpebidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/biassave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
´
save/AssignVariableOp_31AssignVariableOpgbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernelsave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
_output_shapes
:*
T0
ą
save/AssignVariableOp_32AssignVariableOpdbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/biassave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
_output_shapes
:*
T0
ł
save/AssignVariableOp_33AssignVariableOpfbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernelsave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:

save/AssignVariableOp_34AssignVariableOpObidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/biassave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:

save/AssignVariableOp_35AssignVariableOpQbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernelsave/Identity_36*
dtype0
Ą
save/AssignAssign
dense/biassave/RestoreV2:36*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ź
save/Assign_1Assigndense/kernelsave/RestoreV2:37*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@dense/kernel
§
save/Assign_2Assign
embeddingssave/RestoreV2:38*
validate_shape(*
_output_shapes

:J*
use_locking(*
T0*
_class
loc:@embeddings
Ą
save/Assign_3Assignglobal_stepsave/RestoreV2:39*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 

save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9^save/Assign_1^save/Assign_2^save/Assign_3
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"Ó~
trainable_variablesť~¸~
G
embeddings:0embeddings/Assignembeddings/read:02random_normal:08

Sbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"Ĺ
while_context˛Ž

+bidirectional_rnn/fw/fw/while/while_context *(bidirectional_rnn/fw/fw/while/LoopCond:02%bidirectional_rnn/fw/fw/while/Merge:0:(bidirectional_rnn/fw/fw/while/Identity:0B$bidirectional_rnn/fw/fw/while/Exit:0B&bidirectional_rnn/fw/fw/while/Exit_1:0B&bidirectional_rnn/fw/fw/while/Exit_2:0B&bidirectional_rnn/fw/fw/while/Exit_3:0B&bidirectional_rnn/fw/fw/while/Exit_4:0B&bidirectional_rnn/fw/fw/while/Exit_5:0JÖ
%bidirectional_rnn/fw/fw/CheckSeqLen:0
!bidirectional_rnn/fw/fw/Minimum:0
%bidirectional_rnn/fw/fw/TensorArray:0
Tbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
'bidirectional_rnn/fw/fw/TensorArray_1:0
)bidirectional_rnn/fw/fw/strided_slice_1:0
%bidirectional_rnn/fw/fw/while/Enter:0
'bidirectional_rnn/fw/fw/while/Enter_1:0
'bidirectional_rnn/fw/fw/while/Enter_2:0
'bidirectional_rnn/fw/fw/while/Enter_3:0
'bidirectional_rnn/fw/fw/while/Enter_4:0
'bidirectional_rnn/fw/fw/while/Enter_5:0
$bidirectional_rnn/fw/fw/while/Exit:0
&bidirectional_rnn/fw/fw/while/Exit_1:0
&bidirectional_rnn/fw/fw/while/Exit_2:0
&bidirectional_rnn/fw/fw/while/Exit_3:0
&bidirectional_rnn/fw/fw/while/Exit_4:0
&bidirectional_rnn/fw/fw/while/Exit_5:0
2bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0
,bidirectional_rnn/fw/fw/while/GreaterEqual:0
(bidirectional_rnn/fw/fw/while/Identity:0
*bidirectional_rnn/fw/fw/while/Identity_1:0
*bidirectional_rnn/fw/fw/while/Identity_2:0
*bidirectional_rnn/fw/fw/while/Identity_3:0
*bidirectional_rnn/fw/fw/while/Identity_4:0
*bidirectional_rnn/fw/fw/while/Identity_5:0
*bidirectional_rnn/fw/fw/while/Less/Enter:0
$bidirectional_rnn/fw/fw/while/Less:0
,bidirectional_rnn/fw/fw/while/Less_1/Enter:0
&bidirectional_rnn/fw/fw/while/Less_1:0
*bidirectional_rnn/fw/fw/while/LogicalAnd:0
(bidirectional_rnn/fw/fw/while/LoopCond:0
%bidirectional_rnn/fw/fw/while/Merge:0
%bidirectional_rnn/fw/fw/while/Merge:1
'bidirectional_rnn/fw/fw/while/Merge_1:0
'bidirectional_rnn/fw/fw/while/Merge_1:1
'bidirectional_rnn/fw/fw/while/Merge_2:0
'bidirectional_rnn/fw/fw/while/Merge_2:1
'bidirectional_rnn/fw/fw/while/Merge_3:0
'bidirectional_rnn/fw/fw/while/Merge_3:1
'bidirectional_rnn/fw/fw/while/Merge_4:0
'bidirectional_rnn/fw/fw/while/Merge_4:1
'bidirectional_rnn/fw/fw/while/Merge_5:0
'bidirectional_rnn/fw/fw/while/Merge_5:1
-bidirectional_rnn/fw/fw/while/NextIteration:0
/bidirectional_rnn/fw/fw/while/NextIteration_1:0
/bidirectional_rnn/fw/fw/while/NextIteration_2:0
/bidirectional_rnn/fw/fw/while/NextIteration_3:0
/bidirectional_rnn/fw/fw/while/NextIteration_4:0
/bidirectional_rnn/fw/fw/while/NextIteration_5:0
,bidirectional_rnn/fw/fw/while/Select/Enter:0
&bidirectional_rnn/fw/fw/while/Select:0
(bidirectional_rnn/fw/fw/while/Select_1:0
(bidirectional_rnn/fw/fw/while/Select_2:0
(bidirectional_rnn/fw/fw/while/Select_3:0
&bidirectional_rnn/fw/fw/while/Switch:0
&bidirectional_rnn/fw/fw/while/Switch:1
(bidirectional_rnn/fw/fw/while/Switch_1:0
(bidirectional_rnn/fw/fw/while/Switch_1:1
(bidirectional_rnn/fw/fw/while/Switch_2:0
(bidirectional_rnn/fw/fw/while/Switch_2:1
(bidirectional_rnn/fw/fw/while/Switch_3:0
(bidirectional_rnn/fw/fw/while/Switch_3:1
(bidirectional_rnn/fw/fw/while/Switch_4:0
(bidirectional_rnn/fw/fw/while/Switch_4:1
(bidirectional_rnn/fw/fw/while/Switch_5:0
(bidirectional_rnn/fw/fw/while/Switch_5:1
7bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0
9bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0
1bidirectional_rnn/fw/fw/while/TensorArrayReadV3:0
Ibidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Cbidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3:0
%bidirectional_rnn/fw/fw/while/add/y:0
#bidirectional_rnn/fw/fw/while/add:0
'bidirectional_rnn/fw/fw/while/add_1/y:0
%bidirectional_rnn/fw/fw/while/add_1:0
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/Enter:0
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Const:0
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/Enter:0
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul:0
abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/Enter:0
[bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1:0
abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/Enter:0
[bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2:0
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Sigmoid:0
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Tanh:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1:0
^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat/axis:0
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_1:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_2:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split/split_dim:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:1
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub/x:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub:0
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/Enter:0
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Const:0
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/Enter:0
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul:0
abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/Enter:0
[bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1:0
abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/Enter:0
[bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2:0
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Sigmoid:0
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Tanh:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1:0
^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat/axis:0
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_1:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_2:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split/split_dim:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:1
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub/x:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub:0
`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/Enter:0
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0
\bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Const:0
_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/Enter:0
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul:0
abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/Enter:0
[bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1:0
abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/Enter:0
[bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2:0
Zbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Sigmoid:0
Wbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Tanh:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1:0
^bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat/axis:0
Ybidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_1:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_2:0
bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split/split_dim:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:0
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:1
Xbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub/x:0
Vbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub:0
bidirectional_rnn/fw/fw/zeros:0
ubidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0
wbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0
tbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0
vbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0
_bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0
abidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0
ubidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0
wbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0
tbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0
vbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0
_bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0
abidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0
ubidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0
wbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0
tbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0
vbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0
_bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0
abidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0Ú
tbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0Ü
wbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/Enter:0Ű
vbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/Enter:0Ä
abidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/Enter:0Ű
ubidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0r
%bidirectional_rnn/fw/fw/TensorArray:0Ibidirectional_rnn/fw/fw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Ü
wbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/Enter:0Ä
abidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/Enter:0
Tbidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:09bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter_1:0O
bidirectional_rnn/fw/fw/zeros:0,bidirectional_rnn/fw/fw/while/Select/Enter:0Ă
_bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/Enter:0Q
!bidirectional_rnn/fw/fw/Minimum:0,bidirectional_rnn/fw/fw/while/Less_1/Enter:0Ű
vbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/Enter:0b
'bidirectional_rnn/fw/fw/TensorArray_1:07bidirectional_rnn/fw/fw/while/TensorArrayReadV3/Enter:0[
%bidirectional_rnn/fw/fw/CheckSeqLen:02bidirectional_rnn/fw/fw/while/GreaterEqual/Enter:0Ű
ubidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0Ü
wbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/Enter:0Ă
_bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/Enter:0W
)bidirectional_rnn/fw/fw/strided_slice_1:0*bidirectional_rnn/fw/fw/while/Less/Enter:0Ű
ubidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0Ú
tbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0Ű
vbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0abidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/Enter:0Ă
_bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0`bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/Enter:0Ä
abidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0_bidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/Enter:0Ú
tbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0bbidirectional_rnn/fw/fw/while/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0R%bidirectional_rnn/fw/fw/while/Enter:0R'bidirectional_rnn/fw/fw/while/Enter_1:0R'bidirectional_rnn/fw/fw/while/Enter_2:0R'bidirectional_rnn/fw/fw/while/Enter_3:0R'bidirectional_rnn/fw/fw/while/Enter_4:0R'bidirectional_rnn/fw/fw/while/Enter_5:0Z)bidirectional_rnn/fw/fw/strided_slice_1:0

+bidirectional_rnn/bw/bw/while/while_context *(bidirectional_rnn/bw/bw/while/LoopCond:02%bidirectional_rnn/bw/bw/while/Merge:0:(bidirectional_rnn/bw/bw/while/Identity:0B$bidirectional_rnn/bw/bw/while/Exit:0B&bidirectional_rnn/bw/bw/while/Exit_1:0B&bidirectional_rnn/bw/bw/while/Exit_2:0B&bidirectional_rnn/bw/bw/while/Exit_3:0B&bidirectional_rnn/bw/bw/while/Exit_4:0B&bidirectional_rnn/bw/bw/while/Exit_5:0JÖ
%bidirectional_rnn/bw/bw/CheckSeqLen:0
!bidirectional_rnn/bw/bw/Minimum:0
%bidirectional_rnn/bw/bw/TensorArray:0
Tbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
'bidirectional_rnn/bw/bw/TensorArray_1:0
)bidirectional_rnn/bw/bw/strided_slice_1:0
%bidirectional_rnn/bw/bw/while/Enter:0
'bidirectional_rnn/bw/bw/while/Enter_1:0
'bidirectional_rnn/bw/bw/while/Enter_2:0
'bidirectional_rnn/bw/bw/while/Enter_3:0
'bidirectional_rnn/bw/bw/while/Enter_4:0
'bidirectional_rnn/bw/bw/while/Enter_5:0
$bidirectional_rnn/bw/bw/while/Exit:0
&bidirectional_rnn/bw/bw/while/Exit_1:0
&bidirectional_rnn/bw/bw/while/Exit_2:0
&bidirectional_rnn/bw/bw/while/Exit_3:0
&bidirectional_rnn/bw/bw/while/Exit_4:0
&bidirectional_rnn/bw/bw/while/Exit_5:0
2bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0
,bidirectional_rnn/bw/bw/while/GreaterEqual:0
(bidirectional_rnn/bw/bw/while/Identity:0
*bidirectional_rnn/bw/bw/while/Identity_1:0
*bidirectional_rnn/bw/bw/while/Identity_2:0
*bidirectional_rnn/bw/bw/while/Identity_3:0
*bidirectional_rnn/bw/bw/while/Identity_4:0
*bidirectional_rnn/bw/bw/while/Identity_5:0
*bidirectional_rnn/bw/bw/while/Less/Enter:0
$bidirectional_rnn/bw/bw/while/Less:0
,bidirectional_rnn/bw/bw/while/Less_1/Enter:0
&bidirectional_rnn/bw/bw/while/Less_1:0
*bidirectional_rnn/bw/bw/while/LogicalAnd:0
(bidirectional_rnn/bw/bw/while/LoopCond:0
%bidirectional_rnn/bw/bw/while/Merge:0
%bidirectional_rnn/bw/bw/while/Merge:1
'bidirectional_rnn/bw/bw/while/Merge_1:0
'bidirectional_rnn/bw/bw/while/Merge_1:1
'bidirectional_rnn/bw/bw/while/Merge_2:0
'bidirectional_rnn/bw/bw/while/Merge_2:1
'bidirectional_rnn/bw/bw/while/Merge_3:0
'bidirectional_rnn/bw/bw/while/Merge_3:1
'bidirectional_rnn/bw/bw/while/Merge_4:0
'bidirectional_rnn/bw/bw/while/Merge_4:1
'bidirectional_rnn/bw/bw/while/Merge_5:0
'bidirectional_rnn/bw/bw/while/Merge_5:1
-bidirectional_rnn/bw/bw/while/NextIteration:0
/bidirectional_rnn/bw/bw/while/NextIteration_1:0
/bidirectional_rnn/bw/bw/while/NextIteration_2:0
/bidirectional_rnn/bw/bw/while/NextIteration_3:0
/bidirectional_rnn/bw/bw/while/NextIteration_4:0
/bidirectional_rnn/bw/bw/while/NextIteration_5:0
,bidirectional_rnn/bw/bw/while/Select/Enter:0
&bidirectional_rnn/bw/bw/while/Select:0
(bidirectional_rnn/bw/bw/while/Select_1:0
(bidirectional_rnn/bw/bw/while/Select_2:0
(bidirectional_rnn/bw/bw/while/Select_3:0
&bidirectional_rnn/bw/bw/while/Switch:0
&bidirectional_rnn/bw/bw/while/Switch:1
(bidirectional_rnn/bw/bw/while/Switch_1:0
(bidirectional_rnn/bw/bw/while/Switch_1:1
(bidirectional_rnn/bw/bw/while/Switch_2:0
(bidirectional_rnn/bw/bw/while/Switch_2:1
(bidirectional_rnn/bw/bw/while/Switch_3:0
(bidirectional_rnn/bw/bw/while/Switch_3:1
(bidirectional_rnn/bw/bw/while/Switch_4:0
(bidirectional_rnn/bw/bw/while/Switch_4:1
(bidirectional_rnn/bw/bw/while/Switch_5:0
(bidirectional_rnn/bw/bw/while/Switch_5:1
7bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0
9bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0
1bidirectional_rnn/bw/bw/while/TensorArrayReadV3:0
Ibidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
Cbidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3:0
%bidirectional_rnn/bw/bw/while/add/y:0
#bidirectional_rnn/bw/bw/while/add:0
'bidirectional_rnn/bw/bw/while/add_1/y:0
%bidirectional_rnn/bw/bw/while/add_1:0
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/Enter:0
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Const:0
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/Enter:0
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul:0
abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/Enter:0
[bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1:0
abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/Enter:0
[bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2:0
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Sigmoid:0
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/Tanh:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/add_1:0
^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat/axis:0
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/concat:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_1:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/mul_2:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split/split_dim:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/split:1
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub/x:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/sub:0
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/Enter:0
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Const:0
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/Enter:0
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul:0
abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/Enter:0
[bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1:0
abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/Enter:0
[bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2:0
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Sigmoid:0
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/Tanh:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/add_1:0
^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat/axis:0
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/concat:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_1:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/mul_2:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split/split_dim:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/split:1
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub/x:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/sub:0
`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/Enter:0
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0
\bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Const:0
_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/Enter:0
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul:0
abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/Enter:0
[bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1:0
abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/Enter:0
[bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2:0
Zbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Sigmoid:0
Wbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/Tanh:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/add_1:0
^bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat/axis:0
Ybidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/concat:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_1:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/mul_2:0
bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split/split_dim:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:0
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/split:1
Xbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub/x:0
Vbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/sub:0
bidirectional_rnn/bw/bw/zeros:0
ubidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0
wbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0
tbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0
vbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0
_bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0
abidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0
ubidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0
wbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0
tbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0
vbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0
_bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0
abidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0
ubidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0
wbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0
tbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0
vbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0
_bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0
abidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0Ă
_bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd/Enter:0Ú
tbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0Ú
tbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0
Tbidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:09bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter_1:0Ä
abidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul/Enter:0Ü
wbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_2/Enter:0Ă
_bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd/Enter:0Ű
ubidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0Ú
tbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_1/Enter:0Ű
vbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_1/Enter:0Ă
_bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0`bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd/Enter:0r
%bidirectional_rnn/bw/bw/TensorArray:0Ibidirectional_rnn/bw/bw/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Ä
abidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul/Enter:0Ü
wbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_2/Enter:0Ű
vbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/MatMul_1/Enter:0Ű
ubidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0O
bidirectional_rnn/bw/bw/zeros:0,bidirectional_rnn/bw/bw/while/Select/Enter:0Ű
ubidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0bbidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/BiasAdd_2/Enter:0Q
!bidirectional_rnn/bw/bw/Minimum:0,bidirectional_rnn/bw/bw/while/Less_1/Enter:0b
'bidirectional_rnn/bw/bw/TensorArray_1:07bidirectional_rnn/bw/bw/while/TensorArrayReadV3/Enter:0Ü
wbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/MatMul_2/Enter:0[
%bidirectional_rnn/bw/bw/CheckSeqLen:02bidirectional_rnn/bw/bw/while/GreaterEqual/Enter:0Ä
abidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0_bidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul/Enter:0Ű
vbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0abidirectional_rnn/bw/bw/while/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/MatMul_1/Enter:0W
)bidirectional_rnn/bw/bw/strided_slice_1:0*bidirectional_rnn/bw/bw/while/Less/Enter:0R%bidirectional_rnn/bw/bw/while/Enter:0R'bidirectional_rnn/bw/bw/while/Enter_1:0R'bidirectional_rnn/bw/bw/while/Enter_2:0R'bidirectional_rnn/bw/bw/while/Enter_3:0R'bidirectional_rnn/bw/bw/while/Enter_4:0R'bidirectional_rnn/bw/bw/while/Enter_5:0Z)bidirectional_rnn/bw/bw/strided_slice_1:0"Ł
	variables
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
G
embeddings:0embeddings/Assignembeddings/read:02random_normal:08

Sbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/fw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/fw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/fw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/bw/multi_rnn_cell/cell_0/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/bw/multi_rnn_cell/cell_1/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08

Sbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel:0Xbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Assignabidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Read/Identity:0(2nbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/kernel/Initializer/random_uniform:08
ő
Qbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias:0Vbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Assign_bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Read/Identity:0(2cbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/gates/bias/Initializer/Const:08
Ű
hbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel:0mbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Assignvbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/kernel/Initializer/random_uniform:08
ß
ibidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel:0nbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Assignwbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Read/Identity:0(2bidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel/Initializer/random_uniform:08
É
fbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias:0kbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Assigntbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Read/Identity:0(2xbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/input_projection/bias/Initializer/zeros:08
Í
gbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias:0lbidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Assignubidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Read/Identity:0(2ybidirectional_rnn/bw/multi_rnn_cell/cell_2/cudnn_compatible_gru_cell/candidate/hidden_projection/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
V
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:08*
serving_defaultü
'
lengths
	lengths:0	˙˙˙˙˙˙˙˙˙
>
encoded_text.
encoded_text:0	˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙>
probabilities-
	Softmax:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙5
	class_ids(
ArgMax:0	˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙tensorflow/serving/predict