       гK"	   HОК╓Abrain.Event:2ЇоЦR╩     ·ВFЩ	╞╛HОК╓A"┼Ф

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
П
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
У
tensors/component_0Const"/device:CPU:0*=
value4B2
"(l  S  О  K  W  к  ;  ▌  p    *
dtype0*
_output_shapes
:

У
tensors/component_1Const"/device:CPU:0*=
value4B2
"(Й  =    2  └   $  C  H  █  э  *
dtype0*
_output_shapes
:

ч
tensors/component_2Const"/device:CPU:0*Р
valueЖBГ
BSoutheasternBSoutheasternBBig TenBBig 12BAtlantic CoastBAtlantic CoastBPac-12BMountain WestBBig TenBSoutheastern*
dtype0*
_output_shapes
:

У
tensors/component_3Const"/device:CPU:0*=
value4B2
"(▌  ╘  ╤  ╤  ╤  ╧  ╧  ┘  ┘  ╫  *
dtype0*
_output_shapes
:

У
tensors/component_4Const"/device:CPU:0*=
value4B2
"(/   +   -         -      -   .   $   *
dtype0*
_output_shapes
:

У
tensors/component_5Const"/device:CPU:0*=
value4B2
"(                                        *
dtype0*
_output_shapes
:

У
tensors/component_6Const"/device:CPU:0*=
value4B2
"('   #   -                   .      *
dtype0*
_output_shapes
:

У
tensors/component_7Const"/device:CPU:0*=
value4B2
"(э          m            о   ╔      *
dtype0*
_output_shapes
:

У
tensors/component_8Const"/device:CPU:0*=
value4B2
"(                              *
dtype0*
_output_shapes
:

У
tensors/component_9Const"/device:CPU:0*=
value4B2
"(  А   №   д   ы   ╤  л   Д   с   Л   *
dtype0*
_output_shapes
:

Ф
tensors/component_10Const"/device:CPU:0*=
value4B2
"(                              *
dtype0*
_output_shapes
:

Ф
tensors/component_11Const"/device:CPU:0*=
value4B2
"(  y   Д  Ф      o  Ш   \  O   *
dtype0*
_output_shapes
:

Ь
tensors/component_12Const"/device:CPU:0*E
value<B:
BSFOBSDGBSDGBWASBATLBPHIBCINBDENBINDBOAK*
dtype0*
_output_shapes
:

Ф
tensors/component_13Const"/device:CPU:0*=
value4B2
"(4   Q   Z         M   -   /   C   4   *
dtype0*
_output_shapes
:

Ф
tensors/component_14Const"/device:CPU:0*=
value4B2
"(ё   З'  .  D  у  ┼     ╔  Ы+  с  *
dtype0*
_output_shapes
:

Ф
tensors/component_15Const"/device:CPU:0*=
value4B2
"(                                    *
dtype0*
_output_shapes
:

╩
OneShotIteratorOneShotIterator"/device:CPU:0*
_output_shapes
: *-
dataset_factoryR
_make_dataset_ad3a09d5*
shared_name *Е
output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         *
	container *$
output_types
2
∙
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*Е
output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         *Ж
_output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         *$
output_types
2
Н
Bdnn/input_from_feature_columns/input_layer/Attempts/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
▀
>dnn/input_from_feature_columns/input_layer/Attempts/ExpandDims
ExpandDimsIteratorGetNextBdnn/input_from_feature_columns/input_layer/Attempts/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
─
;dnn/input_from_feature_columns/input_layer/Attempts/ToFloatCast>dnn/input_from_feature_columns/input_layer/Attempts/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
┤
9dnn/input_from_feature_columns/input_layer/Attempts/ShapeShape;dnn/input_from_feature_columns/input_layer/Attempts/ToFloat*
T0*
out_type0*
_output_shapes
:
С
Gdnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¤
Adnn/input_from_feature_columns/input_layer/Attempts/strided_sliceStridedSlice9dnn/input_from_feature_columns/input_layer/Attempts/ShapeGdnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stackIdnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_1Idnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Е
Cdnn/input_from_feature_columns/input_layer/Attempts/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Л
Adnn/input_from_feature_columns/input_layer/Attempts/Reshape/shapePackAdnn/input_from_feature_columns/input_layer/Attempts/strided_sliceCdnn/input_from_feature_columns/input_layer/Attempts/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Ж
;dnn/input_from_feature_columns/input_layer/Attempts/ReshapeReshape;dnn/input_from_feature_columns/input_layer/Attempts/ToFloatAdnn/input_from_feature_columns/input_layer/Attempts/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Р
Ednn/input_from_feature_columns/input_layer/Completions/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ч
Adnn/input_from_feature_columns/input_layer/Completions/ExpandDims
ExpandDimsIteratorGetNext:1Ednn/input_from_feature_columns/input_layer/Completions/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╩
>dnn/input_from_feature_columns/input_layer/Completions/ToFloatCastAdnn/input_from_feature_columns/input_layer/Completions/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
║
<dnn/input_from_feature_columns/input_layer/Completions/ShapeShape>dnn/input_from_feature_columns/input_layer/Completions/ToFloat*
T0*
out_type0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/Completions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
М
Ddnn/input_from_feature_columns/input_layer/Completions/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/Completions/ShapeJdnn/input_from_feature_columns/input_layer/Completions/strided_slice/stackLdnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
И
Fdnn/input_from_feature_columns/input_layer/Completions/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ф
Ddnn/input_from_feature_columns/input_layer/Completions/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/Completions/strided_sliceFdnn/input_from_feature_columns/input_layer/Completions/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
П
>dnn/input_from_feature_columns/input_layer/Completions/ReshapeReshape>dnn/input_from_feature_columns/input_layer/Completions/ToFloatDdnn/input_from_feature_columns/input_layer/Completions/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Щ
Ndnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
∙
Jdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims
ExpandDimsIteratorGetNext:2Ndnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
▀
Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/ShapeShapeJdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims*
T0*
out_type0*
_output_shapes
:
ч
Tdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/CastCastUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Shape*

SrcT0*
_output_shapes
:*

DstT0	
Щ
Xdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast_1/xConst*
valueB B *
dtype0*
_output_shapes
: 
╝
Xdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/NotEqualNotEqualJdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDimsXdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
с
Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/WhereWhereXdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         
░
]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╔
Wdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/ReshapeReshapeJdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
┤
cdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stackConst*
valueB"       *
dtype0*
_output_shapes
:
╢
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╢
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ц
]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_sliceStridedSliceUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Wherecdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stackednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_1ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:         
╢
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
╕
gdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╕
gdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
в
_dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1StridedSliceUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Whereednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stackgdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_1gdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_2*
T0	*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:         
ё
Wdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/unstackUnpackTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast*
T0	*	
num*

axis *
_output_shapes
: : 
Є
Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/stackPackYdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/unstack:1*
T0	*

axis *
N*
_output_shapes
:
─
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/MulMul_dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/stack*
T0	*'
_output_shapes
:         
п
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
с
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/SumSumSdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Mulednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0	*#
_output_shapes
:         
╝
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/AddAdd]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_sliceSdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
ъ
Vdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/GatherGatherWdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/ReshapeSdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Add*
Tparams0*
validate_indices(*#
_output_shapes
:         *
Tindices0	
Х
\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_tableHashTableV2*
use_node_name_sharing( *
	key_dtype0*
	container *
value_dtype0	*
_output_shapes
: *>
shared_name/-hash_table_vocab_list/conference.txt_12_-2_-1
н
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/ConstConst*
valueB	 R
         *
dtype0	*
_output_shapes
: 
╨
vdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init/asset_filepathConst**
value!B Bvocab_list/conference.txt*
dtype0*
_output_shapes
: 
╕
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_initInitializeTableFromTextFileV2\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_tablevdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init/asset_filepath*
	delimiter	*
	key_index■        *
value_index         *

vocab_size
╝
Qdnn/input_from_feature_columns/input_layer/Conference_embedding/hash_table_LookupLookupTableFindV2\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_tableVdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Gatherbdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/Const*#
_output_shapes
:         *	
Tin0*

Tout0	
╣
{dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
valueB"      *
dtype0*
_output_shapes
:
м
zdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
о
|dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
valueB
 *:═У>*
dtype0*
_output_shapes
: 
╥
Еdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*

seed *
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
seed2 *
dtype0*
_output_shapes

:
Ь
ydnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulЕdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
_output_shapes

:
Й
udnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
_output_shapes

:
╣
Xdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0
VariableV2*
shared_name *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
∙
_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:*
use_locking(
┘
]dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
_output_shapes

:
▓
hdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
▒
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╞
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SliceSliceTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Casthdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
м
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
я
adnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Const*
T0	*
_output_shapes
: *

Tidx0*
	keep_dims( 
н
kdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
 
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GatherGatherTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Castkdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather/indices*
Tparams0	*
validate_indices(*
_output_shapes
: *
Tindices0
э
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather*
T0	*

axis *
N*
_output_shapes
:
╝
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshapeSparseReshapeUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/WhereTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Castcdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast/x*-
_output_shapes
:         :
А
sdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape/IdentityIdentityQdnn/input_from_feature_columns/input_layer/Conference_embedding/hash_table_Lookup*
T0	*#
_output_shapes
:         
н
kdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Й
idnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
 
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqual*'
_output_shapes
:         
╜
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
√
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:         
б
ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_1Gatherjdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape*
Tparams0	*
validate_indices(*'
_output_shapes
:         *
Tindices0	
ж
ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_2Gathersdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape*
Tparams0	*
validate_indices(*#
_output_shapes
:         *
Tindices0	
Д
ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
╕
vdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
и
Дdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_1ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_2ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
┌
Иdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
▄
Кdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
▄
Кdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
▐
Вdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsИdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stackКdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Кdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*#
_output_shapes
:         
├
ydnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/CastCastВdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:         *

DstT0
┌
{dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*
out_idx0*2
_output_shapes 
:         :         
╣
Еdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/embedding_lookupGather]dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/Unique*
Tparams0*
validate_indices(*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*'
_output_shapes
:         *
Tindices0	
╔
tdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparseSparseSegmentMeanЕdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/embedding_lookup}dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:         
╜
ldnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
и
fdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1ReshapeЖdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
Ц
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
║
pdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
╝
rdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
╝
rdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ж
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
Ў
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice*
T0*

axis *
N*
_output_shapes
:
В
adnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stack*

Tmultiples0*
T0
*0
_output_shapes
:                  
Ь
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┌
\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
ї
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast_1CastTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast*

SrcT0	*
_output_shapes
:*

DstT0
┤
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
│
idnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
█
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
А
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights*
T0*
out_type0*
_output_shapes
:
┤
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
╝
idnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
▄
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
к
hdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
▀
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
Ї
fdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weightscdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:         
ы
Ednn/input_from_feature_columns/input_layer/Conference_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
Э
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Я
Udnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Я
Udnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╣
Mdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/Conference_embedding/ShapeSdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
С
Odnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
п
Mdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
╔
Gdnn/input_from_feature_columns/input_layer/Conference_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
О
Cdnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
?dnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims
ExpandDimsIteratorGetNext:3Cdnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╞
<dnn/input_from_feature_columns/input_layer/DraftYear/ToFloatCast?dnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╢
:dnn/input_from_feature_columns/input_layer/DraftYear/ShapeShape<dnn/input_from_feature_columns/input_layer/DraftYear/ToFloat*
T0*
out_type0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
В
Bdnn/input_from_feature_columns/input_layer/DraftYear/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/DraftYear/ShapeHdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stackJdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ж
Ddnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
О
Bdnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/DraftYear/strided_sliceDdnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Й
<dnn/input_from_feature_columns/input_layer/DraftYear/ReshapeReshape<dnn/input_from_feature_columns/input_layer/DraftYear/ToFloatBdnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Р
Ednn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ч
Adnn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims
ExpandDimsIteratorGetNext:4Ednn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╩
>dnn/input_from_feature_columns/input_layer/GamesPlayed/ToFloatCastAdnn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
║
<dnn/input_from_feature_columns/input_layer/GamesPlayed/ShapeShape>dnn/input_from_feature_columns/input_layer/GamesPlayed/ToFloat*
T0*
out_type0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
М
Ddnn/input_from_feature_columns/input_layer/GamesPlayed/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/GamesPlayed/ShapeJdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stackLdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
И
Fdnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ф
Ddnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_sliceFdnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
П
>dnn/input_from_feature_columns/input_layer/GamesPlayed/ReshapeReshape>dnn/input_from_feature_columns/input_layer/GamesPlayed/ToFloatDdnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
М
Adnn/input_from_feature_columns/input_layer/Heisman/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
▀
=dnn/input_from_feature_columns/input_layer/Heisman/ExpandDims
ExpandDimsIteratorGetNext:5Adnn/input_from_feature_columns/input_layer/Heisman/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
┬
:dnn/input_from_feature_columns/input_layer/Heisman/ToFloatCast=dnn/input_from_feature_columns/input_layer/Heisman/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
▓
8dnn/input_from_feature_columns/input_layer/Heisman/ShapeShape:dnn/input_from_feature_columns/input_layer/Heisman/ToFloat*
T0*
out_type0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
°
@dnn/input_from_feature_columns/input_layer/Heisman/strided_sliceStridedSlice8dnn/input_from_feature_columns/input_layer/Heisman/ShapeFdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stackHdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_1Hdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Д
Bdnn/input_from_feature_columns/input_layer/Heisman/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
И
@dnn/input_from_feature_columns/input_layer/Heisman/Reshape/shapePack@dnn/input_from_feature_columns/input_layer/Heisman/strided_sliceBdnn/input_from_feature_columns/input_layer/Heisman/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Г
:dnn/input_from_feature_columns/input_layer/Heisman/ReshapeReshape:dnn/input_from_feature_columns/input_layer/Heisman/ToFloat@dnn/input_from_feature_columns/input_layer/Heisman/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Т
Gdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ы
Cdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims
ExpandDimsIteratorGetNext:6Gdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╬
@dnn/input_from_feature_columns/input_layer/Interceptions/ToFloatCastCdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╛
>dnn/input_from_feature_columns/input_layer/Interceptions/ShapeShape@dnn/input_from_feature_columns/input_layer/Interceptions/ToFloat*
T0*
out_type0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Fdnn/input_from_feature_columns/input_layer/Interceptions/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/Interceptions/ShapeLdnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stackNdnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
К
Hdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ъ
Fdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/Interceptions/strided_sliceHdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Х
@dnn/input_from_feature_columns/input_layer/Interceptions/ReshapeReshape@dnn/input_from_feature_columns/input_layer/Interceptions/ToFloatFdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Й
>dnn/input_from_feature_columns/input_layer/Pick/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
┘
:dnn/input_from_feature_columns/input_layer/Pick/ExpandDims
ExpandDimsIteratorGetNext:7>dnn/input_from_feature_columns/input_layer/Pick/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╝
7dnn/input_from_feature_columns/input_layer/Pick/ToFloatCast:dnn/input_from_feature_columns/input_layer/Pick/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
м
5dnn/input_from_feature_columns/input_layer/Pick/ShapeShape7dnn/input_from_feature_columns/input_layer/Pick/ToFloat*
T0*
out_type0*
_output_shapes
:
Н
Cdnn/input_from_feature_columns/input_layer/Pick/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
П
Ednn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
П
Ednn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
щ
=dnn/input_from_feature_columns/input_layer/Pick/strided_sliceStridedSlice5dnn/input_from_feature_columns/input_layer/Pick/ShapeCdnn/input_from_feature_columns/input_layer/Pick/strided_slice/stackEdnn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_1Ednn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Б
?dnn/input_from_feature_columns/input_layer/Pick/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
 
=dnn/input_from_feature_columns/input_layer/Pick/Reshape/shapePack=dnn/input_from_feature_columns/input_layer/Pick/strided_slice?dnn/input_from_feature_columns/input_layer/Pick/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
·
7dnn/input_from_feature_columns/input_layer/Pick/ReshapeReshape7dnn/input_from_feature_columns/input_layer/Pick/ToFloat=dnn/input_from_feature_columns/input_layer/Pick/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
К
?dnn/input_from_feature_columns/input_layer/Round/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
;dnn/input_from_feature_columns/input_layer/Round/ExpandDims
ExpandDimsIteratorGetNext:8?dnn/input_from_feature_columns/input_layer/Round/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╛
8dnn/input_from_feature_columns/input_layer/Round/ToFloatCast;dnn/input_from_feature_columns/input_layer/Round/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
о
6dnn/input_from_feature_columns/input_layer/Round/ShapeShape8dnn/input_from_feature_columns/input_layer/Round/ToFloat*
T0*
out_type0*
_output_shapes
:
О
Ddnn/input_from_feature_columns/input_layer/Round/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ю
>dnn/input_from_feature_columns/input_layer/Round/strided_sliceStridedSlice6dnn/input_from_feature_columns/input_layer/Round/ShapeDdnn/input_from_feature_columns/input_layer/Round/strided_slice/stackFdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_1Fdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
В
@dnn/input_from_feature_columns/input_layer/Round/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
В
>dnn/input_from_feature_columns/input_layer/Round/Reshape/shapePack>dnn/input_from_feature_columns/input_layer/Round/strided_slice@dnn/input_from_feature_columns/input_layer/Round/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
¤
8dnn/input_from_feature_columns/input_layer/Round/ReshapeReshape8dnn/input_from_feature_columns/input_layer/Round/ToFloat>dnn/input_from_feature_columns/input_layer/Round/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
С
Fdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
Bdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims
ExpandDimsIteratorGetNext:9Fdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╠
?dnn/input_from_feature_columns/input_layer/RushAttempts/ToFloatCastBdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╝
=dnn/input_from_feature_columns/input_layer/RushAttempts/ShapeShape?dnn/input_from_feature_columns/input_layer/RushAttempts/ToFloat*
T0*
out_type0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
С
Ednn/input_from_feature_columns/input_layer/RushAttempts/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/RushAttempts/ShapeKdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stackMdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Й
Gdnn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ч
Ednn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/RushAttempts/strided_sliceGdnn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Т
?dnn/input_from_feature_columns/input_layer/RushAttempts/ReshapeReshape?dnn/input_from_feature_columns/input_layer/RushAttempts/ToFloatEdnn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
У
Hdnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ю
Ddnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims
ExpandDimsIteratorGetNext:10Hdnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╨
Adnn/input_from_feature_columns/input_layer/RushTouchdowns/ToFloatCastDdnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
└
?dnn/input_from_feature_columns/input_layer/RushTouchdowns/ShapeShapeAdnn/input_from_feature_columns/input_layer/RushTouchdowns/ToFloat*
T0*
out_type0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ы
Gdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/RushTouchdowns/ShapeMdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stackOdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Л
Idnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Э
Gdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_sliceIdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Ш
Adnn/input_from_feature_columns/input_layer/RushTouchdowns/ReshapeReshapeAdnn/input_from_feature_columns/input_layer/RushTouchdowns/ToFloatGdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
О
Cdnn/input_from_feature_columns/input_layer/RushYards/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ф
?dnn/input_from_feature_columns/input_layer/RushYards/ExpandDims
ExpandDimsIteratorGetNext:11Cdnn/input_from_feature_columns/input_layer/RushYards/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╞
<dnn/input_from_feature_columns/input_layer/RushYards/ToFloatCast?dnn/input_from_feature_columns/input_layer/RushYards/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╢
:dnn/input_from_feature_columns/input_layer/RushYards/ShapeShape<dnn/input_from_feature_columns/input_layer/RushYards/ToFloat*
T0*
out_type0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
В
Bdnn/input_from_feature_columns/input_layer/RushYards/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/RushYards/ShapeHdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stackJdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Ж
Ddnn/input_from_feature_columns/input_layer/RushYards/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
О
Bdnn/input_from_feature_columns/input_layer/RushYards/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/RushYards/strided_sliceDdnn/input_from_feature_columns/input_layer/RushYards/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Й
<dnn/input_from_feature_columns/input_layer/RushYards/ReshapeReshape<dnn/input_from_feature_columns/input_layer/RushYards/ToFloatBdnn/input_from_feature_columns/input_layer/RushYards/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
У
Hdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ю
Ddnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims
ExpandDimsIteratorGetNext:12Hdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╙
Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/ShapeShapeDdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims*
T0*
out_type0*
_output_shapes
:
█
Ndnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/CastCastOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Shape*

SrcT0*
_output_shapes
:*

DstT0	
У
Rdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast_1/xConst*
valueB B *
dtype0*
_output_shapes
: 
к
Rdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/NotEqualNotEqualDdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDimsRdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
╒
Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/WhereWhereRdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         
к
Wdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╖
Qdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/ReshapeReshapeDdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDimsWdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
о
]dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stackConst*
valueB"       *
dtype0*
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
°
Wdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_sliceStridedSliceOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Where]dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_1_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:         
░
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Д
Ydnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1StridedSliceOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Where_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stackadnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_1adnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_2*
Index0*
T0	*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:         
х
Qdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/unstackUnpackNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast*
T0	*	
num*

axis *
_output_shapes
: : 
ц
Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/stackPackSdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/unstack:1*
T0	*

axis *
N*
_output_shapes
:
▓
Mdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/MulMulYdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/stack*
T0	*'
_output_shapes
:         
й
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╧
Mdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/SumSumMdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Mul_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Sum/reduction_indices*
T0	*#
_output_shapes
:         *

Tidx0*
	keep_dims( 
к
Mdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/AddAddWdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_sliceMdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
╪
Pdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/GatherGatherQdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/ReshapeMdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Add*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:         
Д
Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_tableHashTableV2*
value_dtype0	*
_output_shapes
: *9
shared_name*(hash_table_vocab_list/teams.txt_32_-2_-1*
use_node_name_sharing( *
	key_dtype0*
	container 
б
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/ConstConst*
valueB	 R
         *
dtype0	*
_output_shapes
: 
┐
jdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init/asset_filepathConst*%
valueB Bvocab_list/teams.txt*
dtype0*
_output_shapes
: 
Ф
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_initInitializeTableFromTextFileV2Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_tablejdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init/asset_filepath*
	key_index■        *
value_index         *

vocab_size *
	delimiter	
Ш
Kdnn/input_from_feature_columns/input_layer/Team_embedding/hash_table_LookupLookupTableFindV2Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_tablePdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/GatherVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:         *	
Tin0
н
udnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
valueB"        *
dtype0*
_output_shapes
:
а
tdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
в
vdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
valueB
 *є5>*
dtype0*
_output_shapes
: 
┐
dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaludnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*

seed *
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
seed2 *
dtype0*
_output_shapes

:  
Г
sdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalvdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
_output_shapes

:  
ё
odnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normalAddsdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/multdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
_output_shapes

:  
н
Rdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0
VariableV2*
shape
:  *
dtype0*
_output_shapes

:  *
shared_name *e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
	container 
с
Ydnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/AssignAssignRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0odnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal*
use_locking(*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:  
╟
Wdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/readIdentityRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
_output_shapes

:  
ж
\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
е
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SliceSliceNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/begin[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
а
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
╦
Udnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ProdProdVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SliceVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Const*
T0	*
_output_shapes
: *

Tidx0*
	keep_dims( 
б
_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GatherGatherNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather/indices*
Tparams0	*
validate_indices(*
_output_shapes
: *
Tindices0
╔
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast/xPackUdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ProdWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather*
T0	*

axis *
N*
_output_shapes
:
Ш
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshapeSparseReshapeOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/WhereNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/CastWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast/x*-
_output_shapes
:         :
ю
gdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape/IdentityIdentityKdnn/input_from_feature_columns/input_layer/Team_embedding/hash_table_Lookup*
T0	*#
_output_shapes
:         
б
_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
х
]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqualGreaterEqualgdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape/Identity_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
ч
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/WhereWhere]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqual*'
_output_shapes
:         
▒
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╫
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ReshapeReshapeVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Where^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:         
¤
Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_1Gather^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshapeXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape*
Tparams0	*
validate_indices(*'
_output_shapes
:         *
Tindices0	
В
Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_2Gathergdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape/IdentityXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape*
Tparams0	*
validate_indices(*#
_output_shapes
:         *
Tindices0	
ь
Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/IdentityIdentity`dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
м
jdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
ы
xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_1Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_2Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Identityjdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
═
|dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
╧
~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╧
~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Э
vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicexdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows|dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:         
к
mdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/CastCastvdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:         *

DstT0
┴
odnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/UniqueUniquezdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*
out_idx0*2
_output_shapes 
:         :         
Ф
ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherWdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/readodnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*
Tparams0*
validate_indices(*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*'
_output_shapes
:          
Ш
hdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparseSparseSegmentMeanydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/embedding_lookupqdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/Unique:1mdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:          
▒
`dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Г
Zdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1Reshapezdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2`dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
■
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ShapeShapehdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
о
ddnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
░
fdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
░
fdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
О
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_sliceStridedSliceVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Shapeddnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stackfdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_1fdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ъ
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
╥
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stackPackXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stack/0^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice*
T0*

axis *
N*
_output_shapes
:
▐
Udnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/TileTileZdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
Д
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/zeros_like	ZerosLikehdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:          
к
Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weightsSelectUdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Tile[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/zeros_likehdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:          
у
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast_1CastNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast*

SrcT0	*
_output_shapes
:*

DstT0
и
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
з
]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
л
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1SliceWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast_1^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/begin]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
ш
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Shape_1ShapePdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights*
T0*
out_type0*
_output_shapes
:
и
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
░
]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
м
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2SliceXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Shape_1^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/begin]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ю
\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
п
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concatConcatV2Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
╨
Zdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_2ReshapePdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weightsWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:          
┘
?dnn/input_from_feature_columns/input_layer/Team_embedding/ShapeShapeZdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ы
Gdnn/input_from_feature_columns/input_layer/Team_embedding/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/Team_embedding/ShapeMdnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stackOdnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Л
Idnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
Э
Gdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/Team_embedding/strided_sliceIdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
▒
Adnn/input_from_feature_columns/input_layer/Team_embedding/ReshapeReshapeZdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_2Gdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:          
П
Ddnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ц
@dnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims
ExpandDimsIteratorGetNext:13Ddnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╚
=dnn/input_from_feature_columns/input_layer/Touchdowns/ToFloatCast@dnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╕
;dnn/input_from_feature_columns/input_layer/Touchdowns/ShapeShape=dnn/input_from_feature_columns/input_layer/Touchdowns/ToFloat*
T0*
out_type0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
Cdnn/input_from_feature_columns/input_layer/Touchdowns/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/Touchdowns/ShapeIdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stackKdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
З
Ednn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
С
Cdnn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/Touchdowns/strided_sliceEdnn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
М
=dnn/input_from_feature_columns/input_layer/Touchdowns/ReshapeReshape=dnn/input_from_feature_columns/input_layer/Touchdowns/ToFloatCdnn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
К
?dnn/input_from_feature_columns/input_layer/Yards/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
▄
;dnn/input_from_feature_columns/input_layer/Yards/ExpandDims
ExpandDimsIteratorGetNext:14?dnn/input_from_feature_columns/input_layer/Yards/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╛
8dnn/input_from_feature_columns/input_layer/Yards/ToFloatCast;dnn/input_from_feature_columns/input_layer/Yards/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
о
6dnn/input_from_feature_columns/input_layer/Yards/ShapeShape8dnn/input_from_feature_columns/input_layer/Yards/ToFloat*
T0*
out_type0*
_output_shapes
:
О
Ddnn/input_from_feature_columns/input_layer/Yards/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ю
>dnn/input_from_feature_columns/input_layer/Yards/strided_sliceStridedSlice6dnn/input_from_feature_columns/input_layer/Yards/ShapeDdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stackFdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_1Fdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
В
@dnn/input_from_feature_columns/input_layer/Yards/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
В
>dnn/input_from_feature_columns/input_layer/Yards/Reshape/shapePack>dnn/input_from_feature_columns/input_layer/Yards/strided_slice@dnn/input_from_feature_columns/input_layer/Yards/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
¤
8dnn/input_from_feature_columns/input_layer/Yards/ReshapeReshape8dnn/input_from_feature_columns/input_layer/Yards/ToFloat>dnn/input_from_feature_columns/input_layer/Yards/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
я
1dnn/input_from_feature_columns/input_layer/concatConcatV2;dnn/input_from_feature_columns/input_layer/Attempts/Reshape>dnn/input_from_feature_columns/input_layer/Completions/ReshapeGdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape<dnn/input_from_feature_columns/input_layer/DraftYear/Reshape>dnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape:dnn/input_from_feature_columns/input_layer/Heisman/Reshape@dnn/input_from_feature_columns/input_layer/Interceptions/Reshape7dnn/input_from_feature_columns/input_layer/Pick/Reshape8dnn/input_from_feature_columns/input_layer/Round/Reshape?dnn/input_from_feature_columns/input_layer/RushAttempts/ReshapeAdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape<dnn/input_from_feature_columns/input_layer/RushYards/ReshapeAdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape=dnn/input_from_feature_columns/input_layer/Touchdowns/Reshape8dnn/input_from_feature_columns/input_layer/Yards/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         9
┼
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB"9   2   *
dtype0*
_output_shapes
:
╖
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *■{r╛*
dtype0*
_output_shapes
: 
╖
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *■{r>*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:92*

seed 
Ъ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
м
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:92
Ю
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:92
╟
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:92*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:92
У
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:92*
use_locking(
о
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:92
о
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
valueB2*    *
dtype0*
_output_shapes
:2
╗
dnn/hiddenlayer_0/bias/part_0
VariableV2*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container *
shape:2*
dtype0*
_output_shapes
:2*
shared_name 
■
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:2*
use_locking(
д
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:2
s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0*
_output_shapes

:92
╟
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
transpose_b( *
T0*'
_output_shapes
:         2*
transpose_a( 
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
T0*
_output_shapes
:2
Я
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:         2
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:         2
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*'
_output_shapes
:         2
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*'
_output_shapes
:         2*

DstT0
h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Н
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
л
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
┼
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB"2   d   *
dtype0*
_output_shapes
:
╖
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *═╠L╛*
dtype0*
_output_shapes
: 
╖
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:2d
Ъ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
м
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:2d
Ю
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:2d
╟
dnn/hiddenlayer_1/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:2d*
dtype0*
_output_shapes

:2d*
shared_name 
У
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:2d*
use_locking(
о
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:2d
о
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
valueBd*    *
dtype0*
_output_shapes
:d
╗
dnn/hiddenlayer_1/bias/part_0
VariableV2*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:d*
dtype0*
_output_shapes
:d
■
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:d*
use_locking(
д
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:d
s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
T0*
_output_shapes

:2d
м
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:         d*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
:d
Я
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*'
_output_shapes
:         d
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:         d
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ж
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:         d
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:         d*

DstT0
j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
У
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
┼
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB"d   2   *
dtype0*
_output_shapes
:
╖
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB
 *═╠L╛*
dtype0*
_output_shapes
: 
╖
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:d2*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
seed2 
Ъ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
м
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:d2
Ю
:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:d2
╟
dnn/hiddenlayer_2/kernel/part_0
VariableV2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
	container *
shape
:d2*
dtype0*
_output_shapes

:d2
У
&dnn/hiddenlayer_2/kernel/part_0/AssignAssigndnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(*
_output_shapes

:d2*
use_locking(
о
$dnn/hiddenlayer_2/kernel/part_0/readIdentitydnn/hiddenlayer_2/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:d2
о
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
valueB2*    *
dtype0*
_output_shapes
:2
╗
dnn/hiddenlayer_2/bias/part_0
VariableV2*
dtype0*
_output_shapes
:2*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
	container *
shape:2
■
$dnn/hiddenlayer_2/bias/part_0/AssignAssigndnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes
:2
д
"dnn/hiddenlayer_2/bias/part_0/readIdentitydnn/hiddenlayer_2/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:2
s
dnn/hiddenlayer_2/kernelIdentity$dnn/hiddenlayer_2/kernel/part_0/read*
T0*
_output_shapes

:d2
м
dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
T0*'
_output_shapes
:         2*
transpose_a( *
transpose_b( 
k
dnn/hiddenlayer_2/biasIdentity"dnn/hiddenlayer_2/bias/part_0/read*
T0*
_output_shapes
:2
Я
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
T0*
data_formatNHWC*'
_output_shapes
:         2
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:         2
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ж
dnn/zero_fraction_2/EqualEqualdnn/hiddenlayer_2/Reludnn/zero_fraction_2/zero*
T0*'
_output_shapes
:         2
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*'
_output_shapes
:         2*

DstT0
j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
У
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
╖
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB"2      *
dtype0*
_output_shapes
:
й
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *SЭп╛*
dtype0*
_output_shapes
: 
й
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *SЭп>*
dtype0*
_output_shapes
: 
Й
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:2*

seed 
■
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
Р
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:2
В
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:2
╣
dnn/logits/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:2*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:2
ў
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:2*
use_locking(
Щ
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:2
а
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
valueB*    *
dtype0*
_output_shapes
:
н
dnn/logits/bias/part_0
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:
т
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
П
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

:2
Ю
dnn/logits/MatMulMatMuldnn/hiddenlayer_2/Reludnn/logits/kernel*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
T0*
_output_shapes
:
К
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*'
_output_shapes
:         
]
dnn/zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:         
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

SrcT0
*'
_output_shapes
:         *

DstT0
j
dnn/zero_fraction_3/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
У
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Т
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Я
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
Б
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
n
,dnn/head/predictions/logits/assert_rank/rankConst*
value	B :*
dtype0*
_output_shapes
: 
^
Vdnn/head/predictions/logits/assert_rank/assert_type/statically_determined_correct_typeNoOp
O
Gdnn/head/predictions/logits/assert_rank/static_checks_determined_all_okNoOp
├
/dnn/head/predictions/logits/strided_slice/stackConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
┼
1dnn/head/predictions/logits/strided_slice/stack_1ConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
┼
1dnn/head/predictions/logits/strided_slice/stack_2ConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
Е
)dnn/head/predictions/logits/strided_sliceStridedSlice!dnn/head/predictions/logits/Shape/dnn/head/predictions/logits/strided_slice/stack1dnn/head/predictions/logits/strided_slice/stack_11dnn/head/predictions/logits/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
╢
*dnn/head/predictions/logits/assert_equal/xConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
п
.dnn/head/predictions/logits/assert_equal/EqualEqual*dnn/head/predictions/logits/assert_equal/x)dnn/head/predictions/logits/strided_slice*
T0*
_output_shapes
: 
╗
.dnn/head/predictions/logits/assert_equal/ConstConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB *
dtype0*
_output_shapes
: 
╚
,dnn/head/predictions/logits/assert_equal/AllAll.dnn/head/predictions/logits/assert_equal/Equal.dnn/head/predictions/logits/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
о
6dnn/head/predictions/logits/assert_equal/Assert/AssertAssert,dnn/head/predictions/logits/assert_equal/All!dnn/head/predictions/logits/Shape*

T
2*
	summarize
Ё
dnn/head/predictions/logitsIdentitydnn/logits/BiasAddH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok7^dnn/head/predictions/logits/assert_equal/Assert/Assert*
T0*'
_output_shapes
:         
w
dnn/head/predictions/logisticSigmoiddnn/head/predictions/logits*
T0*'
_output_shapes
:         
{
dnn/head/predictions/zeros_like	ZerosLikednn/head/predictions/logits*
T0*'
_output_shapes
:         
l
*dnn/head/predictions/two_class_logits/axisConst*
value	B :*
dtype0*
_output_shapes
: 
т
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/head/predictions/logits*dnn/head/predictions/two_class_logits/axis*

Tidx0*
T0*
N*'
_output_shapes
:         
Ж
"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:         
g
%dnn/head/predictions/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
└
dnn/head/predictions/ArgMaxArgMax%dnn/head/predictions/two_class_logits%dnn/head/predictions/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:         *

Tidx0
s
"dnn/head/predictions/classes/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
и
dnn/head/predictions/classesReshapednn/head/predictions/ArgMax"dnn/head/predictions/classes/shape*
T0	*
Tshape0*'
_output_shapes
:         
┌
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/classes*
width         *'
_output_shapes
:         *
	precision         *
shortest( *
T0	*

fill *

scientific( 
s
(dnn/head/maybe_expand_dim/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
о
$dnn/head/maybe_expand_dim/ExpandDims
ExpandDimsIteratorGetNext:15(dnn/head/maybe_expand_dim/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
y
dnn/head/labels/ShapeShape$dnn/head/maybe_expand_dim/ExpandDims*
T0*
out_type0*
_output_shapes
:
b
 dnn/head/labels/assert_rank/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Е
!dnn/head/labels/assert_rank/ShapeShape$dnn/head/maybe_expand_dim/ExpandDims*
T0*
out_type0*
_output_shapes
:
R
Jdnn/head/labels/assert_rank/assert_type/statically_determined_correct_typeNoOp
C
;dnn/head/labels/assert_rank/static_checks_determined_all_okNoOp
л
#dnn/head/labels/strided_slice/stackConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
н
%dnn/head/labels/strided_slice/stack_1Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
н
%dnn/head/labels/strided_slice/stack_2Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
╔
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Ю
dnn/head/labels/assert_equal/xConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
Л
"dnn/head/labels/assert_equal/EqualEqualdnn/head/labels/assert_equal/xdnn/head/labels/strided_slice*
T0*
_output_shapes
: 
г
"dnn/head/labels/assert_equal/ConstConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB *
dtype0*
_output_shapes
: 
д
 dnn/head/labels/assert_equal/AllAll"dnn/head/labels/assert_equal/Equal"dnn/head/labels/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
╠
)dnn/head/labels/assert_equal/Assert/ConstConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*5
value,B* B$labels shape must be [batch_size, 1]*
dtype0*
_output_shapes
: 
╒
+dnn/head/labels/assert_equal/Assert/Const_1Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
╤
+dnn/head/labels/assert_equal/Assert/Const_2Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*8
value/B- B'x (dnn/head/labels/assert_equal/x:0) = *
dtype0*
_output_shapes
: 
╨
+dnn/head/labels/assert_equal/Assert/Const_3Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*7
value.B, B&y (dnn/head/labels/strided_slice:0) = *
dtype0*
_output_shapes
: 
╘
1dnn/head/labels/assert_equal/Assert/Assert/data_0Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*5
value,B* B$labels shape must be [batch_size, 1]*
dtype0*
_output_shapes
: 
█
1dnn/head/labels/assert_equal/Assert/Assert/data_1Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
╫
1dnn/head/labels/assert_equal/Assert/Assert/data_2Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*8
value/B- B'x (dnn/head/labels/assert_equal/x:0) = *
dtype0*
_output_shapes
: 
╓
1dnn/head/labels/assert_equal/Assert/Assert/data_4Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*7
value.B, B&y (dnn/head/labels/strided_slice:0) = *
dtype0*
_output_shapes
: 
Г
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_01dnn/head/labels/assert_equal/Assert/Assert/data_11dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/assert_equal/x1dnn/head/labels/assert_equal/Assert/Assert/data_4dnn/head/labels/strided_slice*
T

2*
	summarize
▐
dnn/head/labelsIdentity$dnn/head/maybe_expand_dim/ExpandDims<^dnn/head/labels/assert_rank/static_checks_determined_all_ok+^dnn/head/labels/assert_equal/Assert/Assert*
T0*'
_output_shapes
:         
j
dnn/head/ToFloatCastdnn/head/labels*

SrcT0*'
_output_shapes
:         *

DstT0
`
dnn/head/assert_range/ConstConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
П
&dnn/head/assert_range/assert_less/LessLessdnn/head/ToFloatdnn/head/assert_range/Const*
T0*'
_output_shapes
:         
x
'dnn/head/assert_range/assert_less/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
▓
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
Й
.dnn/head/assert_range/assert_less/Assert/ConstConst*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
┤
0dnn/head/assert_range/assert_less/Assert/Const_1Const*T
valueKBI BCCondition x < y did not hold element-wise:x (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
Х
0dnn/head/assert_range/assert_less/Assert/Const_2Const*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
╢
;dnn/head/assert_range/assert_less/Assert/AssertGuard/SwitchSwitch%dnn/head/assert_range/assert_less/All%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: : 
й
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
з
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_fIdentity;dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
Р
<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_idIdentity%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: 
Б
9dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOpNoOp>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
┴
Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependencyIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:^dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOp*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t*
_output_shapes
: 
▌
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
Ж
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*T
valueKBI BCCondition x < y did not hold element-wise:x (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
ч
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_3Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
О
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchSwitch%dnn/head/assert_range/assert_less/All<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0
*8
_class.
,*loc:@dnn/head/assert_range/assert_less/All*
_output_shapes
: : 
И
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloat<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0*#
_class
loc:@dnn/head/ToFloat*:
_output_shapes(
&:         :         
№
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0*.
_class$
" loc:@dnn/head/assert_range/Const*
_output_shapes
: : 
В
;dnn/head/assert_range/assert_less/Assert/AssertGuard/AssertAssertBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_3Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
┼
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
_output_shapes
: 
Г
:dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeMergeIdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
t
/dnn/head/assert_range/assert_non_negative/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╟
Ednn/head/assert_range/assert_non_negative/assert_less_equal/LessEqual	LessEqual/dnn/head/assert_range/assert_non_negative/Constdnn/head/ToFloat*
T0*'
_output_shapes
:         
Т
Adnn/head/assert_range/assert_non_negative/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
Ь
Hdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/ConstConst*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
╢
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_1Const*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
д
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_2Const**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
Д
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/All?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: : 
▌
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
█
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
─
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: 
╡
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
й
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
К
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
в
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
Р
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
Ў
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
╝
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloatVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0*#
_class
loc:@dnn/head/ToFloat*:
_output_shapes(
&:         :         
╫
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
T
2*
	summarize
н
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
╤
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
В
dnn/head/assert_range/IdentityIdentitydnn/head/ToFloat;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*
T0*'
_output_shapes
:         
}
!dnn/head/logistic_loss/zeros_like	ZerosLikednn/head/predictions/logits*
T0*'
_output_shapes
:         
е
#dnn/head/logistic_loss/GreaterEqualGreaterEqualdnn/head/predictions/logits!dnn/head/logistic_loss/zeros_like*
T0*'
_output_shapes
:         
╛
dnn/head/logistic_loss/SelectSelect#dnn/head/logistic_loss/GreaterEqualdnn/head/predictions/logits!dnn/head/logistic_loss/zeros_like*
T0*'
_output_shapes
:         
p
dnn/head/logistic_loss/NegNegdnn/head/predictions/logits*
T0*'
_output_shapes
:         
╣
dnn/head/logistic_loss/Select_1Select#dnn/head/logistic_loss/GreaterEqualdnn/head/logistic_loss/Negdnn/head/predictions/logits*
T0*'
_output_shapes
:         
Р
dnn/head/logistic_loss/mulMuldnn/head/predictions/logitsdnn/head/assert_range/Identity*
T0*'
_output_shapes
:         
О
dnn/head/logistic_loss/subSubdnn/head/logistic_loss/Selectdnn/head/logistic_loss/mul*
T0*'
_output_shapes
:         
t
dnn/head/logistic_loss/ExpExpdnn/head/logistic_loss/Select_1*
T0*'
_output_shapes
:         
s
dnn/head/logistic_loss/Log1pLog1pdnn/head/logistic_loss/Exp*
T0*'
_output_shapes
:         
Й
dnn/head/logistic_lossAdddnn/head/logistic_loss/subdnn/head/logistic_loss/Log1p*
T0*'
_output_shapes
:         
x
3dnn/head/weighted_loss/assert_broadcastable/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
|
9dnn/head/weighted_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
z
8dnn/head/weighted_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
О
8dnn/head/weighted_loss/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
T0*
out_type0*
_output_shapes
:
y
7dnn/head/weighted_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
O
Gdnn/head/weighted_loss/assert_broadcastable/static_scalar_check_successNoOp
▒
"dnn/head/weighted_loss/ToFloat_1/xConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
П
dnn/head/weighted_loss/MulMuldnn/head/logistic_loss"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:         
╖
dnn/head/weighted_loss/ConstConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
Щ
dnn/head/weighted_loss/SumSumdnn/head/weighted_loss/Muldnn/head/weighted_loss/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
z
5dnn/head/metrics/label/mean/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
Pdnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
С
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
н
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
Р
Ndnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
f
^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
№
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ShapeShapednn/head/assert_range/Identity_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
у
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ConstConst_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
я
7dnn/head/metrics/label/mean/broadcast_weights/ones_likeFill=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Shape=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
╓
-dnn/head/metrics/label/mean/broadcast_weightsMul5dnn/head/metrics/label/mean/broadcast_weights/weights7dnn/head/metrics/label/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
о
3dnn/head/metrics/label/mean/total/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
!dnn/head/metrics/label/mean/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
	container *
shape: 
К
(dnn/head/metrics/label/mean/total/AssignAssign!dnn/head/metrics/label/mean/total3dnn/head/metrics/label/mean/total/Initializer/zeros*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
validate_shape(*
_output_shapes
: *
use_locking(
м
&dnn/head/metrics/label/mean/total/readIdentity!dnn/head/metrics/label/mean/total*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: 
о
3dnn/head/metrics/label/mean/count/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
!dnn/head/metrics/label/mean/count
VariableV2*
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
К
(dnn/head/metrics/label/mean/count/AssignAssign!dnn/head/metrics/label/mean/count3dnn/head/metrics/label/mean/count/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
validate_shape(*
_output_shapes
: 
м
&dnn/head/metrics/label/mean/count/readIdentity!dnn/head/metrics/label/mean/count*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: 
┐
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape-dnn/head/metrics/label/mean/broadcast_weights*
T0*
out_type0*
_output_shapes
:
У
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
п
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
Т
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Т
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
Э
Ndnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
з
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
ч
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentity\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
х
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityZdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
╪
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Щ
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
├
Бdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*c
_classY
WUloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
╟
Гdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
о
zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualБdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchГdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Щ
tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Ы
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityvdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Щ
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitytdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
Ю
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
╥
Нdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
▄
Рdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
╕
Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchРdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Ї
Йdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsФdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Нdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
┘
Оdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╩
Оdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
▄
Иdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillОdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeОdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
╞
Кdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ї
Еdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Йdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsИdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeКdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
╘
Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
р
Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╜
Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchТdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
·
Лdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЦdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:*

Tdim0
┴
Чdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationЛdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Еdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
х
Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeЩdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
╝
Аdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
╜
~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualАdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xПdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
и
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*П
_classД
Бloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
а
sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergevdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
у
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergesdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
▓
Jdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ы
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╝
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
Ъ
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
н
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
Ч
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
║
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
с
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
▀
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityWdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
р
Xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
╣
Udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
▒
cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tV^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
в
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Й
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
к
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
И
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ы
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
Е
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
о
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
к
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
и
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Ъ
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
╠
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssert^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
╡
ednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fX^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
╫
Vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Ў
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/assert_range/IdentityW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
▌
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ConstConstW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ї
9dnn/head/metrics/label/mean/broadcast_weights_1/ones_likeFill?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Shape?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
╥
/dnn/head/metrics/label/mean/broadcast_weights_1Mul-dnn/head/metrics/label/mean/broadcast_weights9dnn/head/metrics/label/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
й
dnn/head/metrics/label/mean/MulMuldnn/head/assert_range/Identity/dnn/head/metrics/label/mean/broadcast_weights_1*
T0*'
_output_shapes
:         
r
!dnn/head/metrics/label/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╕
dnn/head/metrics/label/mean/SumSum/dnn/head/metrics/label/mean/broadcast_weights_1!dnn/head/metrics/label/mean/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
#dnn/head/metrics/label/mean/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
м
!dnn/head/metrics/label/mean/Sum_1Sumdnn/head/metrics/label/mean/Mul#dnn/head/metrics/label/mean/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
т
%dnn/head/metrics/label/mean/AssignAdd	AssignAdd!dnn/head/metrics/label/mean/total!dnn/head/metrics/label/mean/Sum_1*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: *
use_locking( 
Д
'dnn/head/metrics/label/mean/AssignAdd_1	AssignAdd!dnn/head/metrics/label/mean/countdnn/head/metrics/label/mean/Sum ^dnn/head/metrics/label/mean/Mul*
use_locking( *
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: 
j
%dnn/head/metrics/label/mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ю
#dnn/head/metrics/label/mean/GreaterGreater&dnn/head/metrics/label/mean/count/read%dnn/head/metrics/label/mean/Greater/y*
T0*
_output_shapes
: 
Я
#dnn/head/metrics/label/mean/truedivRealDiv&dnn/head/metrics/label/mean/total/read&dnn/head/metrics/label/mean/count/read*
T0*
_output_shapes
: 
h
#dnn/head/metrics/label/mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
!dnn/head/metrics/label/mean/valueSelect#dnn/head/metrics/label/mean/Greater#dnn/head/metrics/label/mean/truediv#dnn/head/metrics/label/mean/value/e*
T0*
_output_shapes
: 
l
'dnn/head/metrics/label/mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
г
%dnn/head/metrics/label/mean/Greater_1Greater'dnn/head/metrics/label/mean/AssignAdd_1'dnn/head/metrics/label/mean/Greater_1/y*
T0*
_output_shapes
: 
б
%dnn/head/metrics/label/mean/truediv_1RealDiv%dnn/head/metrics/label/mean/AssignAdd'dnn/head/metrics/label/mean/AssignAdd_1*
T0*
_output_shapes
: 
l
'dnn/head/metrics/label/mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╟
%dnn/head/metrics/label/mean/update_opSelect%dnn/head/metrics/label/mean/Greater_1%dnn/head/metrics/label/mean/truediv_1'dnn/head/metrics/label/mean/update_op/e*
T0*
_output_shapes
: 
▓
5dnn/head/metrics/average_loss/total/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
#dnn/head/metrics/average_loss/total
VariableV2*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Т
*dnn/head/metrics/average_loss/total/AssignAssign#dnn/head/metrics/average_loss/total5dnn/head/metrics/average_loss/total/Initializer/zeros*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
validate_shape(*
_output_shapes
: *
use_locking(
▓
(dnn/head/metrics/average_loss/total/readIdentity#dnn/head/metrics/average_loss/total*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
▓
5dnn/head/metrics/average_loss/count/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
#dnn/head/metrics/average_loss/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
	container *
shape: 
Т
*dnn/head/metrics/average_loss/count/AssignAssign#dnn/head/metrics/average_loss/count5dnn/head/metrics/average_loss/count/Initializer/zeros*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
validate_shape(*
_output_shapes
: *
use_locking(
▓
(dnn/head/metrics/average_loss/count/readIdentity#dnn/head/metrics/average_loss/count*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
h
#dnn/head/metrics/average_loss/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
Rdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
У
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
з
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
T0*
out_type0*
_output_shapes
:
Т
Pdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
h
`dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
°
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ShapeShapednn/head/logistic_lossa^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ч
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ConstConsta^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ї
9dnn/head/metrics/average_loss/broadcast_weights/ones_likeFill?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Shape?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
╚
/dnn/head/metrics/average_loss/broadcast_weightsMul#dnn/head/metrics/average_loss/Const9dnn/head/metrics/average_loss/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
г
!dnn/head/metrics/average_loss/MulMuldnn/head/logistic_loss/dnn/head/metrics/average_loss/broadcast_weights*
T0*'
_output_shapes
:         
v
%dnn/head/metrics/average_loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
╛
!dnn/head/metrics/average_loss/SumSum/dnn/head/metrics/average_loss/broadcast_weights%dnn/head/metrics/average_loss/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
v
%dnn/head/metrics/average_loss/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
▓
#dnn/head/metrics/average_loss/Sum_1Sum!dnn/head/metrics/average_loss/Mul%dnn/head/metrics/average_loss/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ъ
'dnn/head/metrics/average_loss/AssignAdd	AssignAdd#dnn/head/metrics/average_loss/total#dnn/head/metrics/average_loss/Sum_1*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
О
)dnn/head/metrics/average_loss/AssignAdd_1	AssignAdd#dnn/head/metrics/average_loss/count!dnn/head/metrics/average_loss/Sum"^dnn/head/metrics/average_loss/Mul*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
l
'dnn/head/metrics/average_loss/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
д
%dnn/head/metrics/average_loss/GreaterGreater(dnn/head/metrics/average_loss/count/read'dnn/head/metrics/average_loss/Greater/y*
T0*
_output_shapes
: 
е
%dnn/head/metrics/average_loss/truedivRealDiv(dnn/head/metrics/average_loss/total/read(dnn/head/metrics/average_loss/count/read*
T0*
_output_shapes
: 
j
%dnn/head/metrics/average_loss/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
├
#dnn/head/metrics/average_loss/valueSelect%dnn/head/metrics/average_loss/Greater%dnn/head/metrics/average_loss/truediv%dnn/head/metrics/average_loss/value/e*
T0*
_output_shapes
: 
n
)dnn/head/metrics/average_loss/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
й
'dnn/head/metrics/average_loss/Greater_1Greater)dnn/head/metrics/average_loss/AssignAdd_1)dnn/head/metrics/average_loss/Greater_1/y*
T0*
_output_shapes
: 
з
'dnn/head/metrics/average_loss/truediv_1RealDiv'dnn/head/metrics/average_loss/AssignAdd)dnn/head/metrics/average_loss/AssignAdd_1*
T0*
_output_shapes
: 
n
)dnn/head/metrics/average_loss/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╧
'dnn/head/metrics/average_loss/update_opSelect'dnn/head/metrics/average_loss/Greater_1'dnn/head/metrics/average_loss/truediv_1)dnn/head/metrics/average_loss/update_op/e*
T0*
_output_shapes
: 
[
dnn/head/metrics/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
|
dnn/head/metrics/CastCastdnn/head/predictions/classes*

SrcT0	*'
_output_shapes
:         *

DstT0
И
dnn/head/metrics/EqualEqualdnn/head/metrics/Castdnn/head/assert_range/Identity*
T0*'
_output_shapes
:         
y
dnn/head/metrics/ToFloatCastdnn/head/metrics/Equal*

SrcT0
*'
_output_shapes
:         *

DstT0
к
1dnn/head/metrics/accuracy/total/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
valueB
 *    *
dtype0*
_output_shapes
: 
╖
dnn/head/metrics/accuracy/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
	container *
shape: 
В
&dnn/head/metrics/accuracy/total/AssignAssigndnn/head/metrics/accuracy/total1dnn/head/metrics/accuracy/total/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
validate_shape(*
_output_shapes
: 
ж
$dnn/head/metrics/accuracy/total/readIdentitydnn/head/metrics/accuracy/total*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
к
1dnn/head/metrics/accuracy/count/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
valueB
 *    *
dtype0*
_output_shapes
: 
╖
dnn/head/metrics/accuracy/count
VariableV2*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
В
&dnn/head/metrics/accuracy/count/AssignAssigndnn/head/metrics/accuracy/count1dnn/head/metrics/accuracy/count/Initializer/zeros*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
validate_shape(*
_output_shapes
: *
use_locking(
ж
$dnn/head/metrics/accuracy/count/readIdentitydnn/head/metrics/accuracy/count*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
С
Ndnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
П
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
е
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/metrics/ToFloat*
T0*
out_type0*
_output_shapes
:
О
Ldnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
d
\dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Є
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ShapeShapednn/head/metrics/ToFloat]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
▀
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ConstConst]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
щ
5dnn/head/metrics/accuracy/broadcast_weights/ones_likeFill;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Shape;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
│
+dnn/head/metrics/accuracy/broadcast_weightsMuldnn/head/metrics/Const5dnn/head/metrics/accuracy/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
Э
dnn/head/metrics/accuracy/MulMuldnn/head/metrics/ToFloat+dnn/head/metrics/accuracy/broadcast_weights*
T0*'
_output_shapes
:         
p
dnn/head/metrics/accuracy/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
░
dnn/head/metrics/accuracy/SumSum+dnn/head/metrics/accuracy/broadcast_weightsdnn/head/metrics/accuracy/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
r
!dnn/head/metrics/accuracy/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
ж
dnn/head/metrics/accuracy/Sum_1Sumdnn/head/metrics/accuracy/Mul!dnn/head/metrics/accuracy/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
┌
#dnn/head/metrics/accuracy/AssignAdd	AssignAdddnn/head/metrics/accuracy/totaldnn/head/metrics/accuracy/Sum_1*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: *
use_locking( 
·
%dnn/head/metrics/accuracy/AssignAdd_1	AssignAdddnn/head/metrics/accuracy/countdnn/head/metrics/accuracy/Sum^dnn/head/metrics/accuracy/Mul*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: *
use_locking( 
h
#dnn/head/metrics/accuracy/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
!dnn/head/metrics/accuracy/GreaterGreater$dnn/head/metrics/accuracy/count/read#dnn/head/metrics/accuracy/Greater/y*
T0*
_output_shapes
: 
Щ
!dnn/head/metrics/accuracy/truedivRealDiv$dnn/head/metrics/accuracy/total/read$dnn/head/metrics/accuracy/count/read*
T0*
_output_shapes
: 
f
!dnn/head/metrics/accuracy/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
│
dnn/head/metrics/accuracy/valueSelect!dnn/head/metrics/accuracy/Greater!dnn/head/metrics/accuracy/truediv!dnn/head/metrics/accuracy/value/e*
T0*
_output_shapes
: 
j
%dnn/head/metrics/accuracy/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
#dnn/head/metrics/accuracy/Greater_1Greater%dnn/head/metrics/accuracy/AssignAdd_1%dnn/head/metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
Ы
#dnn/head/metrics/accuracy/truediv_1RealDiv#dnn/head/metrics/accuracy/AssignAdd%dnn/head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
j
%dnn/head/metrics/accuracy/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
#dnn/head/metrics/accuracy/update_opSelect#dnn/head/metrics/accuracy/Greater_1#dnn/head/metrics/accuracy/truediv_1%dnn/head/metrics/accuracy/update_op/e*
T0*
_output_shapes
: 

:dnn/head/metrics/prediction/mean/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ш
Udnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ц
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
▒
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Х
Sdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
k
cdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Е
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
э
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ConstConstd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
■
<dnn/head/metrics/prediction/mean/broadcast_weights/ones_likeFillBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
х
2dnn/head/metrics/prediction/mean/broadcast_weightsMul:dnn/head/metrics/prediction/mean/broadcast_weights/weights<dnn/head/metrics/prediction/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
╕
8dnn/head/metrics/prediction/mean/total/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
┼
&dnn/head/metrics/prediction/mean/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
	container 
Ю
-dnn/head/metrics/prediction/mean/total/AssignAssign&dnn/head/metrics/prediction/mean/total8dnn/head/metrics/prediction/mean/total/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
validate_shape(*
_output_shapes
: 
╗
+dnn/head/metrics/prediction/mean/total/readIdentity&dnn/head/metrics/prediction/mean/total*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
_output_shapes
: 
╕
8dnn/head/metrics/prediction/mean/count/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
┼
&dnn/head/metrics/prediction/mean/count
VariableV2*
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
Ю
-dnn/head/metrics/prediction/mean/count/AssignAssign&dnn/head/metrics/prediction/mean/count8dnn/head/metrics/prediction/mean/count/Initializer/zeros*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
validate_shape(*
_output_shapes
: *
use_locking(
╗
+dnn/head/metrics/prediction/mean/count/readIdentity&dnn/head/metrics/prediction/mean/count*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
_output_shapes
: 
╔
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape2dnn/head/metrics/prediction/mean/broadcast_weights*
T0*
out_type0*
_output_shapes
:
Ш
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
│
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Ч
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Ч
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
м
Sdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
╢
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
ё
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityadnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
я
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentity_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
т
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentitySdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
н
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
╫
Жdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*h
_class^
\Zloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
█
Иdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
╜
dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualЖdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchИdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
и
ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
е
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentity{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
г
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
и
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
▄
Тdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Ё
Хdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
╠
Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchХdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Г
Оdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЩdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Тdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
у
Уdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╘
Уdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ы
Нdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillУdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeУdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
╨
Пdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
И
Кdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Оdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsНdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeПdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
▐
Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Ї
Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╤
Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЧdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
Й
Рdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЫdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
╨
Ьdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationРdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Кdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
я
Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeЮdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
╞
Еdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
═
Гdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualЕdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xФdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
╜
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Х
_classК
ЗДloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
░
xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMerge{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Гdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Є
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergexdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergecdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
╖
Odnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
а
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╞
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
Я
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
▒
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
Ь
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╔
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
ы
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
щ
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentity\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
ъ
]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
├
Zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
┼
hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t[^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
м
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
У
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╣
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
Т
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
д
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
П
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
┬
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
╛
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╝
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
о
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
Г	
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
╔
jdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f]^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
ц
[dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergejdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
 
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistic\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
ч
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ConstConst\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Д
>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_likeFillDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
с
4dnn/head/metrics/prediction/mean/broadcast_weights_1Mul2dnn/head/metrics/prediction/mean/broadcast_weights>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
▓
$dnn/head/metrics/prediction/mean/MulMuldnn/head/predictions/logistic4dnn/head/metrics/prediction/mean/broadcast_weights_1*
T0*'
_output_shapes
:         
w
&dnn/head/metrics/prediction/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╟
$dnn/head/metrics/prediction/mean/SumSum4dnn/head/metrics/prediction/mean/broadcast_weights_1&dnn/head/metrics/prediction/mean/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
y
(dnn/head/metrics/prediction/mean/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
╗
&dnn/head/metrics/prediction/mean/Sum_1Sum$dnn/head/metrics/prediction/mean/Mul(dnn/head/metrics/prediction/mean/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ў
*dnn/head/metrics/prediction/mean/AssignAdd	AssignAdd&dnn/head/metrics/prediction/mean/total&dnn/head/metrics/prediction/mean/Sum_1*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
_output_shapes
: *
use_locking( 
Э
,dnn/head/metrics/prediction/mean/AssignAdd_1	AssignAdd&dnn/head/metrics/prediction/mean/count$dnn/head/metrics/prediction/mean/Sum%^dnn/head/metrics/prediction/mean/Mul*
use_locking( *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
_output_shapes
: 
o
*dnn/head/metrics/prediction/mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
н
(dnn/head/metrics/prediction/mean/GreaterGreater+dnn/head/metrics/prediction/mean/count/read*dnn/head/metrics/prediction/mean/Greater/y*
T0*
_output_shapes
: 
о
(dnn/head/metrics/prediction/mean/truedivRealDiv+dnn/head/metrics/prediction/mean/total/read+dnn/head/metrics/prediction/mean/count/read*
T0*
_output_shapes
: 
m
(dnn/head/metrics/prediction/mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╧
&dnn/head/metrics/prediction/mean/valueSelect(dnn/head/metrics/prediction/mean/Greater(dnn/head/metrics/prediction/mean/truediv(dnn/head/metrics/prediction/mean/value/e*
T0*
_output_shapes
: 
q
,dnn/head/metrics/prediction/mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▓
*dnn/head/metrics/prediction/mean/Greater_1Greater,dnn/head/metrics/prediction/mean/AssignAdd_1,dnn/head/metrics/prediction/mean/Greater_1/y*
T0*
_output_shapes
: 
░
*dnn/head/metrics/prediction/mean/truediv_1RealDiv*dnn/head/metrics/prediction/mean/AssignAdd,dnn/head/metrics/prediction/mean/AssignAdd_1*
T0*
_output_shapes
: 
q
,dnn/head/metrics/prediction/mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
█
*dnn/head/metrics/prediction/mean/update_opSelect*dnn/head/metrics/prediction/mean/Greater_1*dnn/head/metrics/prediction/mean/truediv_1,dnn/head/metrics/prediction/mean/update_op/e*
T0*
_output_shapes
: 
m
(dnn/head/metrics/accuracy_baseline/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ы
&dnn/head/metrics/accuracy_baseline/subSub(dnn/head/metrics/accuracy_baseline/sub/x!dnn/head/metrics/label/mean/value*
T0*
_output_shapes
: 
Я
(dnn/head/metrics/accuracy_baseline/valueMaximum!dnn/head/metrics/label/mean/value&dnn/head/metrics/accuracy_baseline/sub*
T0*
_output_shapes
: 
o
*dnn/head/metrics/accuracy_baseline/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
г
(dnn/head/metrics/accuracy_baseline/sub_1Sub*dnn/head/metrics/accuracy_baseline/sub_1/x%dnn/head/metrics/label/mean/update_op*
T0*
_output_shapes
: 
й
,dnn/head/metrics/accuracy_baseline/update_opMaximum%dnn/head/metrics/label/mean/update_op(dnn/head/metrics/accuracy_baseline/sub_1*
T0*
_output_shapes
: 
В
dnn/head/metrics/auc/CastCastdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:         *

DstT0

s
.dnn/head/metrics/auc/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
М
Idnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
К
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
е
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Й
Gdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
_
Wdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
э
6dnn/head/metrics/auc/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
╒
6dnn/head/metrics/auc/broadcast_weights/ones_like/ConstConstX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
┌
0dnn/head/metrics/auc/broadcast_weights/ones_likeFill6dnn/head/metrics/auc/broadcast_weights/ones_like/Shape6dnn/head/metrics/auc/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
┴
&dnn/head/metrics/auc/broadcast_weightsMul.dnn/head/metrics/auc/broadcast_weights/weights0dnn/head/metrics/auc/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
b
dnn/head/metrics/auc/Cast_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╢
6dnn/head/metrics/auc/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast_1/x*
T0*'
_output_shapes
:         
А
/dnn/head/metrics/auc/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╥
-dnn/head/metrics/auc/assert_greater_equal/AllAll6dnn/head/metrics/auc/assert_greater_equal/GreaterEqual/dnn/head/metrics/auc/assert_greater_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
Ф
6dnn/head/metrics/auc/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╩
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_1Const*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
Я
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_2Const*7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
╬
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/All-dnn/head/metrics/auc/assert_greater_equal/All*
T0
*
_output_shapes
: : 
╣
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╖
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fIdentityCdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
а
Ddnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_idIdentity-dnn/head/metrics/auc/assert_greater_equal/All*
T0
*
_output_shapes
: 
С
Adnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOpNoOpF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t
с
Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tB^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
Ё
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
д
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
∙
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
о
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/AllDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*@
_class6
42loc:@dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: : 
▓
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
Р
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast_1/xDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/metrics/auc/Cast_1/x*
_output_shapes
: : 
║
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/AssertAssertJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
х
Qdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fD^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
Ы
Bdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/MergeMergeQdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
b
dnn/head/metrics/auc/Cast_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
н
0dnn/head/metrics/auc/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast_2/x*
T0*'
_output_shapes
:         
}
,dnn/head/metrics/auc/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╞
*dnn/head/metrics/auc/assert_less_equal/AllAll0dnn/head/metrics/auc/assert_less_equal/LessEqual,dnn/head/metrics/auc/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
С
3dnn/head/metrics/auc/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╟
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_1Const*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
Ь
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_2Const*7
value.B, B&y (dnn/head/metrics/auc/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
┼
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/All*dnn/head/metrics/auc/assert_less_equal/All*
T0
*
_output_shapes
: : 
│
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_tIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
▒
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fIdentity@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ъ
Adnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_idIdentity*dnn/head/metrics/auc/assert_less_equal/All*
T0
*
_output_shapes
: 
Л
>dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOpNoOpC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t
╒
Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t?^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
ъ
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
Ю
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
є
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*7
value.B, B&y (dnn/head/metrics/auc/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
в
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/AllAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*=
_class3
1/loc:@dnn/head/metrics/auc/assert_less_equal/All*
_output_shapes
: : 
м
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
К
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast_2/xAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/metrics/auc/Cast_2/x*
_output_shapes
: : 
е
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/AssertAssertGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
┘
Ndnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fA^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert*
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
Т
?dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/MergeMergeNdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
s
"dnn/head/metrics/auc/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
к
dnn/head/metrics/auc/ReshapeReshapednn/head/predictions/logistic"dnn/head/metrics/auc/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
u
$dnn/head/metrics/auc/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
к
dnn/head/metrics/auc/Reshape_1Reshapednn/head/metrics/auc/Cast$dnn/head/metrics/auc/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
v
dnn/head/metrics/auc/ShapeShapednn/head/metrics/auc/Reshape*
T0*
out_type0*
_output_shapes
:
r
(dnn/head/metrics/auc/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
т
"dnn/head/metrics/auc/strided_sliceStridedSlicednn/head/metrics/auc/Shape(dnn/head/metrics/auc/strided_slice/stack*dnn/head/metrics/auc/strided_slice/stack_1*dnn/head/metrics/auc/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Й
dnn/head/metrics/auc/ConstConst*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
dtype0*
_output_shapes	
:╚
m
#dnn/head/metrics/auc/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
д
dnn/head/metrics/auc/ExpandDims
ExpandDimsdnn/head/metrics/auc/Const#dnn/head/metrics/auc/ExpandDims/dim*
T0*
_output_shapes
:	╚*

Tdim0
^
dnn/head/metrics/auc/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
Ю
dnn/head/metrics/auc/stackPackdnn/head/metrics/auc/stack/0"dnn/head/metrics/auc/strided_slice*
T0*

axis *
N*
_output_shapes
:
г
dnn/head/metrics/auc/TileTilednn/head/metrics/auc/ExpandDimsdnn/head/metrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
j
#dnn/head/metrics/auc/transpose/RankRankdnn/head/metrics/auc/Reshape*
T0*
_output_shapes
: 
f
$dnn/head/metrics/auc/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Х
"dnn/head/metrics/auc/transpose/subSub#dnn/head/metrics/auc/transpose/Rank$dnn/head/metrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
l
*dnn/head/metrics/auc/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
l
*dnn/head/metrics/auc/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╥
$dnn/head/metrics/auc/transpose/RangeRange*dnn/head/metrics/auc/transpose/Range/start#dnn/head/metrics/auc/transpose/Rank*dnn/head/metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:
Ъ
$dnn/head/metrics/auc/transpose/sub_1Sub"dnn/head/metrics/auc/transpose/sub$dnn/head/metrics/auc/transpose/Range*
T0*
_output_shapes
:
о
dnn/head/metrics/auc/transpose	Transposednn/head/metrics/auc/Reshape$dnn/head/metrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
v
%dnn/head/metrics/auc/Tile_1/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
п
dnn/head/metrics/auc/Tile_1Tilednn/head/metrics/auc/transpose%dnn/head/metrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
Т
dnn/head/metrics/auc/GreaterGreaterdnn/head/metrics/auc/Tile_1dnn/head/metrics/auc/Tile*
T0*(
_output_shapes
:╚         
u
dnn/head/metrics/auc/LogicalNot
LogicalNotdnn/head/metrics/auc/Greater*(
_output_shapes
:╚         
v
%dnn/head/metrics/auc/Tile_2/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
п
dnn/head/metrics/auc/Tile_2Tilednn/head/metrics/auc/Reshape_1%dnn/head/metrics/auc/Tile_2/multiples*
T0
*(
_output_shapes
:╚         *

Tmultiples0
v
!dnn/head/metrics/auc/LogicalNot_1
LogicalNotdnn/head/metrics/auc/Tile_2*(
_output_shapes
:╚         
▒
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeShape&dnn/head/metrics/auc/broadcast_weights*
T0*
out_type0*
_output_shapes
:
М
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
з
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Л
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Л
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
И
Gdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarEqualIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
Т
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
┘
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityUdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
╫
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentitySdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
╩
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
¤
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
ж
zdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*\
_classR
PNloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
к
|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
Ч
sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualzdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Д
mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranksdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Н
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Л
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitymdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
Р
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitysdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
─
Жdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
└
Йdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Ь
Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЙdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
▀
Вdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsНdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Жdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
╦
Зdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╝
Зdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╟
Бdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЗdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЗdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
╕
Гdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╫
~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Вdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsБdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeГdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
╞
Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
─
Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
б
Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЛdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
х
Дdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsПdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
л
Рdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationДdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
╫
Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeТdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
н
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
з
wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xИdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
К
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Ж
_class|
zxloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Л
ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
╬
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergeldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
л
Cdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ф
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
о
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_2Const*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
У
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
е
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
Р
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
е
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
╙
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╤
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityPdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
╥
Qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
л
Ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
Х
\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tO^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
Ф
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
√
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Х
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
·
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
М
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
ў
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Т
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
О
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
М
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
■
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
 
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
Щ
^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fQ^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
┬
Odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMerge^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
ч
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logisticP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
╧
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ConstConstP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
р
2dnn/head/metrics/auc/broadcast_weights_1/ones_likeFill8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Shape8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
╜
(dnn/head/metrics/auc/broadcast_weights_1Mul&dnn/head/metrics/auc/broadcast_weights2dnn/head/metrics/auc/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
u
$dnn/head/metrics/auc/Reshape_2/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
╣
dnn/head/metrics/auc/Reshape_2Reshape(dnn/head/metrics/auc/broadcast_weights_1$dnn/head/metrics/auc/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
v
%dnn/head/metrics/auc/Tile_3/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
п
dnn/head/metrics/auc/Tile_3Tilednn/head/metrics/auc/Reshape_2%dnn/head/metrics/auc/Tile_3/multiples*
T0*(
_output_shapes
:╚         *

Tmultiples0
╝
5dnn/head/metrics/auc/true_positives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╔
#dnn/head/metrics/auc/true_positives
VariableV2*
dtype0*
_output_shapes	
:╚*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
	container *
shape:╚
Ч
*dnn/head/metrics/auc/true_positives/AssignAssign#dnn/head/metrics/auc/true_positives5dnn/head/metrics/auc/true_positives/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
validate_shape(*
_output_shapes	
:╚
╖
(dnn/head/metrics/auc/true_positives/readIdentity#dnn/head/metrics/auc/true_positives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
_output_shapes	
:╚
Т
dnn/head/metrics/auc/LogicalAnd
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/Greater*(
_output_shapes
:╚         
Й
dnn/head/metrics/auc/ToFloat_2Castdnn/head/metrics/auc/LogicalAnd*

SrcT0
*(
_output_shapes
:╚         *

DstT0
П
dnn/head/metrics/auc/mulMuldnn/head/metrics/auc/ToFloat_2dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
l
*dnn/head/metrics/auc/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
и
dnn/head/metrics/auc/SumSumdnn/head/metrics/auc/mul*dnn/head/metrics/auc/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
█
dnn/head/metrics/auc/AssignAdd	AssignAdd#dnn/head/metrics/auc/true_positivesdnn/head/metrics/auc/Sum*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
_output_shapes	
:╚*
use_locking( 
╛
6dnn/head/metrics/auc/false_negatives/Initializer/zerosConst*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╦
$dnn/head/metrics/auc/false_negatives
VariableV2*
shared_name *7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚
Ы
+dnn/head/metrics/auc/false_negatives/AssignAssign$dnn/head/metrics/auc/false_negatives6dnn/head/metrics/auc/false_negatives/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
validate_shape(*
_output_shapes	
:╚
║
)dnn/head/metrics/auc/false_negatives/readIdentity$dnn/head/metrics/auc/false_negatives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:╚
Ч
!dnn/head/metrics/auc/LogicalAnd_1
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
Л
dnn/head/metrics/auc/ToFloat_3Cast!dnn/head/metrics/auc/LogicalAnd_1*

SrcT0
*(
_output_shapes
:╚         *

DstT0
С
dnn/head/metrics/auc/mul_1Muldnn/head/metrics/auc/ToFloat_3dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
n
,dnn/head/metrics/auc/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
о
dnn/head/metrics/auc/Sum_1Sumdnn/head/metrics/auc/mul_1,dnn/head/metrics/auc/Sum_1/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
с
 dnn/head/metrics/auc/AssignAdd_1	AssignAdd$dnn/head/metrics/auc/false_negativesdnn/head/metrics/auc/Sum_1*
use_locking( *
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:╚
╝
5dnn/head/metrics/auc/true_negatives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╔
#dnn/head/metrics/auc/true_negatives
VariableV2*
dtype0*
_output_shapes	
:╚*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
	container *
shape:╚
Ч
*dnn/head/metrics/auc/true_negatives/AssignAssign#dnn/head/metrics/auc/true_negatives5dnn/head/metrics/auc/true_negatives/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
validate_shape(*
_output_shapes	
:╚
╖
(dnn/head/metrics/auc/true_negatives/readIdentity#dnn/head/metrics/auc/true_negatives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
_output_shapes	
:╚
Э
!dnn/head/metrics/auc/LogicalAnd_2
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
Л
dnn/head/metrics/auc/ToFloat_4Cast!dnn/head/metrics/auc/LogicalAnd_2*

SrcT0
*(
_output_shapes
:╚         *

DstT0
С
dnn/head/metrics/auc/mul_2Muldnn/head/metrics/auc/ToFloat_4dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
n
,dnn/head/metrics/auc/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
о
dnn/head/metrics/auc/Sum_2Sumdnn/head/metrics/auc/mul_2,dnn/head/metrics/auc/Sum_2/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
▀
 dnn/head/metrics/auc/AssignAdd_2	AssignAdd#dnn/head/metrics/auc/true_negativesdnn/head/metrics/auc/Sum_2*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
_output_shapes	
:╚
╛
6dnn/head/metrics/auc/false_positives/Initializer/zerosConst*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╦
$dnn/head/metrics/auc/false_positives
VariableV2*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name 
Ы
+dnn/head/metrics/auc/false_positives/AssignAssign$dnn/head/metrics/auc/false_positives6dnn/head/metrics/auc/false_positives/Initializer/zeros*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
validate_shape(*
_output_shapes	
:╚*
use_locking(
║
)dnn/head/metrics/auc/false_positives/readIdentity$dnn/head/metrics/auc/false_positives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:╚
Ъ
!dnn/head/metrics/auc/LogicalAnd_3
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/Greater*(
_output_shapes
:╚         
Л
dnn/head/metrics/auc/ToFloat_5Cast!dnn/head/metrics/auc/LogicalAnd_3*

SrcT0
*(
_output_shapes
:╚         *

DstT0
С
dnn/head/metrics/auc/mul_3Muldnn/head/metrics/auc/ToFloat_5dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
n
,dnn/head/metrics/auc/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
о
dnn/head/metrics/auc/Sum_3Sumdnn/head/metrics/auc/mul_3,dnn/head/metrics/auc/Sum_3/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
с
 dnn/head/metrics/auc/AssignAdd_3	AssignAdd$dnn/head/metrics/auc/false_positivesdnn/head/metrics/auc/Sum_3*
use_locking( *
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:╚
_
dnn/head/metrics/auc/add/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Л
dnn/head/metrics/auc/addAdd(dnn/head/metrics/auc/true_positives/readdnn/head/metrics/auc/add/y*
T0*
_output_shapes	
:╚
Ь
dnn/head/metrics/auc/add_1Add(dnn/head/metrics/auc/true_positives/read)dnn/head/metrics/auc/false_negatives/read*
T0*
_output_shapes	
:╚
a
dnn/head/metrics/auc/add_2/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Б
dnn/head/metrics/auc/add_2Adddnn/head/metrics/auc/add_1dnn/head/metrics/auc/add_2/y*
T0*
_output_shapes	
:╚

dnn/head/metrics/auc/divRealDivdnn/head/metrics/auc/adddnn/head/metrics/auc/add_2*
T0*
_output_shapes	
:╚
Ь
dnn/head/metrics/auc/add_3Add)dnn/head/metrics/auc/false_positives/read(dnn/head/metrics/auc/true_negatives/read*
T0*
_output_shapes	
:╚
a
dnn/head/metrics/auc/add_4/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Б
dnn/head/metrics/auc/add_4Adddnn/head/metrics/auc/add_3dnn/head/metrics/auc/add_4/y*
T0*
_output_shapes	
:╚
Т
dnn/head/metrics/auc/div_1RealDiv)dnn/head/metrics/auc/false_positives/readdnn/head/metrics/auc/add_4*
T0*
_output_shapes	
:╚
t
*dnn/head/metrics/auc/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_1/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_1StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_1/stack,dnn/head/metrics/auc/strided_slice_1/stack_1,dnn/head/metrics/auc/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_2StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_2/stack,dnn/head/metrics/auc/strided_slice_2/stack_1,dnn/head/metrics/auc/strided_slice_2/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟*
Index0*
T0*
shrink_axis_mask 
С
dnn/head/metrics/auc/subSub$dnn/head/metrics/auc/strided_slice_1$dnn/head/metrics/auc/strided_slice_2*
T0*
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_3/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
$dnn/head/metrics/auc/strided_slice_3StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_3/stack,dnn/head/metrics/auc/strided_slice_3/stack_1,dnn/head/metrics/auc/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
T0*
Index0
t
*dnn/head/metrics/auc/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
$dnn/head/metrics/auc/strided_slice_4StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_4/stack,dnn/head/metrics/auc/strided_slice_4/stack_1,dnn/head/metrics/auc/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
У
dnn/head/metrics/auc/add_5Add$dnn/head/metrics/auc/strided_slice_3$dnn/head/metrics/auc/strided_slice_4*
T0*
_output_shapes	
:╟
c
dnn/head/metrics/auc/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Й
dnn/head/metrics/auc/truedivRealDivdnn/head/metrics/auc/add_5dnn/head/metrics/auc/truediv/y*
T0*
_output_shapes	
:╟
}
dnn/head/metrics/auc/MulMuldnn/head/metrics/auc/subdnn/head/metrics/auc/truediv*
T0*
_output_shapes	
:╟
f
dnn/head/metrics/auc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ч
dnn/head/metrics/auc/valueSumdnn/head/metrics/auc/Muldnn/head/metrics/auc/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
a
dnn/head/metrics/auc/add_6/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Е
dnn/head/metrics/auc/add_6Adddnn/head/metrics/auc/AssignAdddnn/head/metrics/auc/add_6/y*
T0*
_output_shapes	
:╚
Й
dnn/head/metrics/auc/add_7Adddnn/head/metrics/auc/AssignAdd dnn/head/metrics/auc/AssignAdd_1*
T0*
_output_shapes	
:╚
a
dnn/head/metrics/auc/add_8/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Б
dnn/head/metrics/auc/add_8Adddnn/head/metrics/auc/add_7dnn/head/metrics/auc/add_8/y*
T0*
_output_shapes	
:╚
Г
dnn/head/metrics/auc/div_2RealDivdnn/head/metrics/auc/add_6dnn/head/metrics/auc/add_8*
T0*
_output_shapes	
:╚
Л
dnn/head/metrics/auc/add_9Add dnn/head/metrics/auc/AssignAdd_3 dnn/head/metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:╚
b
dnn/head/metrics/auc/add_10/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Г
dnn/head/metrics/auc/add_10Adddnn/head/metrics/auc/add_9dnn/head/metrics/auc/add_10/y*
T0*
_output_shapes	
:╚
К
dnn/head/metrics/auc/div_3RealDiv dnn/head/metrics/auc/AssignAdd_3dnn/head/metrics/auc/add_10*
T0*
_output_shapes	
:╚
t
*dnn/head/metrics/auc/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_5/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_5StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_5/stack,dnn/head/metrics/auc/strided_slice_5/stack_1,dnn/head/metrics/auc/strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_6StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_6/stack,dnn/head/metrics/auc/strided_slice_6/stack_1,dnn/head/metrics/auc/strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
У
dnn/head/metrics/auc/sub_1Sub$dnn/head/metrics/auc/strided_slice_5$dnn/head/metrics/auc/strided_slice_6*
T0*
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_7/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_7StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_7/stack,dnn/head/metrics/auc/strided_slice_7/stack_1,dnn/head/metrics/auc/strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_8StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_8/stack,dnn/head/metrics/auc/strided_slice_8/stack_1,dnn/head/metrics/auc/strided_slice_8/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
Ф
dnn/head/metrics/auc/add_11Add$dnn/head/metrics/auc/strided_slice_7$dnn/head/metrics/auc/strided_slice_8*
T0*
_output_shapes	
:╟
e
 dnn/head/metrics/auc/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
О
dnn/head/metrics/auc/truediv_1RealDivdnn/head/metrics/auc/add_11 dnn/head/metrics/auc/truediv_1/y*
T0*
_output_shapes	
:╟
Г
dnn/head/metrics/auc/Mul_1Muldnn/head/metrics/auc/sub_1dnn/head/metrics/auc/truediv_1*
T0*
_output_shapes	
:╟
f
dnn/head/metrics/auc/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Э
dnn/head/metrics/auc/update_opSumdnn/head/metrics/auc/Mul_1dnn/head/metrics/auc/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
У
*dnn/head/metrics/auc_precision_recall/CastCastdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:         *

DstT0

Д
?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
Zdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ы
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
╢
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Ъ
Xdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
p
hdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
П
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logistici^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ў
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ConstConsti^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Н
Adnn/head/metrics/auc_precision_recall/broadcast_weights/ones_likeFillGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
Ї
7dnn/head/metrics/auc_precision_recall/broadcast_weightsMul?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsAdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
s
.dnn/head/metrics/auc_precision_recall/Cast_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╪
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logistic.dnn/head/metrics/auc_precision_recall/Cast_1/x*
T0*'
_output_shapes
:         
С
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllAllGdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqual@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
е
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
█
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_1Const*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
┴
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_2Const*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
Б
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
T0
*
_output_shapes
: : 
█
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
┘
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fIdentityTdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
┬
Udnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_idIdentity>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
T0
*
_output_shapes
: 
│
Rdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOpNoOpW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t
е
`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tS^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
Т
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╞
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
м
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
Є
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*Q
_classG
ECloc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
_output_shapes
: : 
╘
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
╘
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switch.dnn/head/metrics/auc_precision_recall/Cast_1/xUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*A
_class7
53loc:@dnn/head/metrics/auc_precision_recall/Cast_1/x*
_output_shapes
: : 
▒
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/AssertAssert[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
й
bdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fU^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
╬
Sdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/MergeMergebdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
s
.dnn/head/metrics/auc_precision_recall/Cast_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╧
Adnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logistic.dnn/head/metrics/auc_precision_recall/Cast_2/x*
T0*'
_output_shapes
:         
О
=dnn/head/metrics/auc_precision_recall/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
∙
;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllAllAdnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual=dnn/head/metrics/auc_precision_recall/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
в
Ddnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╪
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_1Const*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
╛
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_2Const*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
°
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/All;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: : 
╒
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╙
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fIdentityQdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
╝
Rdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_idIdentity;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: 
н
Odnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOpNoOpT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t
Щ
]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependencyIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tP^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
М
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
└
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
ж
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
ц
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*N
_classD
B@loc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
_output_shapes
: : 
╬
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
╬
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switch.dnn/head/metrics/auc_precision_recall/Cast_2/xRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0*A
_class7
53loc:@dnn/head/metrics/auc_precision_recall/Cast_2/x*
_output_shapes
: : 
Ь
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/AssertAssertXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
Э
_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fR^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
┼
Pdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/MergeMerge_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Д
3dnn/head/metrics/auc_precision_recall/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
╠
-dnn/head/metrics/auc_precision_recall/ReshapeReshapednn/head/predictions/logistic3dnn/head/metrics/auc_precision_recall/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Ж
5dnn/head/metrics/auc_precision_recall/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
▌
/dnn/head/metrics/auc_precision_recall/Reshape_1Reshape*dnn/head/metrics/auc_precision_recall/Cast5dnn/head/metrics/auc_precision_recall/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
Ш
+dnn/head/metrics/auc_precision_recall/ShapeShape-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
out_type0*
_output_shapes
:
Г
9dnn/head/metrics/auc_precision_recall/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Е
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╖
3dnn/head/metrics/auc_precision_recall/strided_sliceStridedSlice+dnn/head/metrics/auc_precision_recall/Shape9dnn/head/metrics/auc_precision_recall/strided_slice/stack;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ъ
+dnn/head/metrics/auc_precision_recall/ConstConst*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
dtype0*
_output_shapes	
:╚
~
4dnn/head/metrics/auc_precision_recall/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
╫
0dnn/head/metrics/auc_precision_recall/ExpandDims
ExpandDims+dnn/head/metrics/auc_precision_recall/Const4dnn/head/metrics/auc_precision_recall/ExpandDims/dim*
T0*
_output_shapes
:	╚*

Tdim0
o
-dnn/head/metrics/auc_precision_recall/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
╤
+dnn/head/metrics/auc_precision_recall/stackPack-dnn/head/metrics/auc_precision_recall/stack/03dnn/head/metrics/auc_precision_recall/strided_slice*
T0*

axis *
N*
_output_shapes
:
╓
*dnn/head/metrics/auc_precision_recall/TileTile0dnn/head/metrics/auc_precision_recall/ExpandDims+dnn/head/metrics/auc_precision_recall/stack*
T0*(
_output_shapes
:╚         *

Tmultiples0
М
4dnn/head/metrics/auc_precision_recall/transpose/RankRank-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
_output_shapes
: 
w
5dnn/head/metrics/auc_precision_recall/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
╚
3dnn/head/metrics/auc_precision_recall/transpose/subSub4dnn/head/metrics/auc_precision_recall/transpose/Rank5dnn/head/metrics/auc_precision_recall/transpose/sub/y*
T0*
_output_shapes
: 
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ц
5dnn/head/metrics/auc_precision_recall/transpose/RangeRange;dnn/head/metrics/auc_precision_recall/transpose/Range/start4dnn/head/metrics/auc_precision_recall/transpose/Rank;dnn/head/metrics/auc_precision_recall/transpose/Range/delta*

Tidx0*
_output_shapes
:
═
5dnn/head/metrics/auc_precision_recall/transpose/sub_1Sub3dnn/head/metrics/auc_precision_recall/transpose/sub5dnn/head/metrics/auc_precision_recall/transpose/Range*
T0*
_output_shapes
:
с
/dnn/head/metrics/auc_precision_recall/transpose	Transpose-dnn/head/metrics/auc_precision_recall/Reshape5dnn/head/metrics/auc_precision_recall/transpose/sub_1*
T0*'
_output_shapes
:         *
Tperm0
З
6dnn/head/metrics/auc_precision_recall/Tile_1/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
т
,dnn/head/metrics/auc_precision_recall/Tile_1Tile/dnn/head/metrics/auc_precision_recall/transpose6dnn/head/metrics/auc_precision_recall/Tile_1/multiples*
T0*(
_output_shapes
:╚         *

Tmultiples0
┼
-dnn/head/metrics/auc_precision_recall/GreaterGreater,dnn/head/metrics/auc_precision_recall/Tile_1*dnn/head/metrics/auc_precision_recall/Tile*
T0*(
_output_shapes
:╚         
Ч
0dnn/head/metrics/auc_precision_recall/LogicalNot
LogicalNot-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:╚         
З
6dnn/head/metrics/auc_precision_recall/Tile_2/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
т
,dnn/head/metrics/auc_precision_recall/Tile_2Tile/dnn/head/metrics/auc_precision_recall/Reshape_16dnn/head/metrics/auc_precision_recall/Tile_2/multiples*
T0
*(
_output_shapes
:╚         *

Tmultiples0
Ш
2dnn/head/metrics/auc_precision_recall/LogicalNot_1
LogicalNot,dnn/head/metrics/auc_precision_recall/Tile_2*(
_output_shapes
:╚         
╙
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeShape7dnn/head/metrics/auc_precision_recall/broadcast_weights*
T0*
out_type0*
_output_shapes
:
Э
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
╕
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Ь
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
╗
Xdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarEqualZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/x[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
┼
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
√
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityfdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
∙
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
ь
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
┴
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
ы
Лdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*m
_classc
a_loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
я
Нdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
═
Дdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualЛdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchНdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
╣
~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
▒
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityАdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
о
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentity~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
│
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
ч
Чdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Д
Ъdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
р
Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЪdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Т
Уdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЮdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Чdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
ю
Шdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
▀
Шdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
·
Тdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillШdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeШdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
█
Фdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ь
Пdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Уdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsТdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeФdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
щ
Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
И
Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
х
Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЬdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
Ш
Хdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsаdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
▀
бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationХdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Пdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
∙
Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeгdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
╤
Кdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
▄
Иdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualКdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xЩdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
╙
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1SwitchДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Ъ
_classП
МЙloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
└
}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeАdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Иdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Б
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMerge}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergehdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
╝
Tdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
е
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╨
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_2Const*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
д
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
╢
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
б
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╪
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergecdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
ї
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
є
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityadnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ї
bdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
═
_dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
┘
mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t`^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
╢
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Э
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╚
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
Ь
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
о
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
Щ
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╓
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
╥
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Switch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╨
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
┬
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarbdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
║	
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAsserthdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchhdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
▌
odnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fb^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
ї
`dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeodnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Й
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistica^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
ё
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ConstConsta^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
Cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_likeFillIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
Ё
9dnn/head/metrics/auc_precision_recall/broadcast_weights_1Mul7dnn/head/metrics/auc_precision_recall/broadcast_weightsCdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
Ж
5dnn/head/metrics/auc_precision_recall/Reshape_2/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
ь
/dnn/head/metrics/auc_precision_recall/Reshape_2Reshape9dnn/head/metrics/auc_precision_recall/broadcast_weights_15dnn/head/metrics/auc_precision_recall/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
З
6dnn/head/metrics/auc_precision_recall/Tile_3/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
т
,dnn/head/metrics/auc_precision_recall/Tile_3Tile/dnn/head/metrics/auc_precision_recall/Reshape_26dnn/head/metrics/auc_precision_recall/Tile_3/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
▐
Fdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
ы
4dnn/head/metrics/auc_precision_recall/true_positives
VariableV2*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚
█
;dnn/head/metrics/auc_precision_recall/true_positives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_positivesFdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
validate_shape(*
_output_shapes	
:╚
ъ
9dnn/head/metrics/auc_precision_recall/true_positives/readIdentity4dnn/head/metrics/auc_precision_recall/true_positives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:╚
┼
0dnn/head/metrics/auc_precision_recall/LogicalAnd
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_2-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:╚         
л
/dnn/head/metrics/auc_precision_recall/ToFloat_2Cast0dnn/head/metrics/auc_precision_recall/LogicalAnd*

SrcT0
*(
_output_shapes
:╚         *

DstT0
┬
)dnn/head/metrics/auc_precision_recall/mulMul/dnn/head/metrics/auc_precision_recall/ToFloat_2,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         
}
;dnn/head/metrics/auc_precision_recall/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
█
)dnn/head/metrics/auc_precision_recall/SumSum)dnn/head/metrics/auc_precision_recall/mul;dnn/head/metrics/auc_precision_recall/Sum/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
Я
/dnn/head/metrics/auc_precision_recall/AssignAdd	AssignAdd4dnn/head/metrics/auc_precision_recall/true_positives)dnn/head/metrics/auc_precision_recall/Sum*
use_locking( *
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:╚
р
Gdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
э
5dnn/head/metrics/auc_precision_recall/false_negatives
VariableV2*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name 
▀
<dnn/head/metrics/auc_precision_recall/false_negatives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_negativesGdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
validate_shape(*
_output_shapes	
:╚
э
:dnn/head/metrics/auc_precision_recall/false_negatives/readIdentity5dnn/head/metrics/auc_precision_recall/false_negatives*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:╚
╩
2dnn/head/metrics/auc_precision_recall/LogicalAnd_1
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_20dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:╚         
н
/dnn/head/metrics/auc_precision_recall/ToFloat_3Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_1*

SrcT0
*(
_output_shapes
:╚         *

DstT0
─
+dnn/head/metrics/auc_precision_recall/mul_1Mul/dnn/head/metrics/auc_precision_recall/ToFloat_3,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         

=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
+dnn/head/metrics/auc_precision_recall/Sum_1Sum+dnn/head/metrics/auc_precision_recall/mul_1=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
е
1dnn/head/metrics/auc_precision_recall/AssignAdd_1	AssignAdd5dnn/head/metrics/auc_precision_recall/false_negatives+dnn/head/metrics/auc_precision_recall/Sum_1*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:╚*
use_locking( 
▐
Fdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
ы
4dnn/head/metrics/auc_precision_recall/true_negatives
VariableV2*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚
█
;dnn/head/metrics/auc_precision_recall/true_negatives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_negativesFdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
validate_shape(*
_output_shapes	
:╚
ъ
9dnn/head/metrics/auc_precision_recall/true_negatives/readIdentity4dnn/head/metrics/auc_precision_recall/true_negatives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:╚
╨
2dnn/head/metrics/auc_precision_recall/LogicalAnd_2
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_10dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:╚         
н
/dnn/head/metrics/auc_precision_recall/ToFloat_4Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_2*

SrcT0
*(
_output_shapes
:╚         *

DstT0
─
+dnn/head/metrics/auc_precision_recall/mul_2Mul/dnn/head/metrics/auc_precision_recall/ToFloat_4,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         

=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
+dnn/head/metrics/auc_precision_recall/Sum_2Sum+dnn/head/metrics/auc_precision_recall/mul_2=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
г
1dnn/head/metrics/auc_precision_recall/AssignAdd_2	AssignAdd4dnn/head/metrics/auc_precision_recall/true_negatives+dnn/head/metrics/auc_precision_recall/Sum_2*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:╚*
use_locking( 
р
Gdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
э
5dnn/head/metrics/auc_precision_recall/false_positives
VariableV2*
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
	container 
▀
<dnn/head/metrics/auc_precision_recall/false_positives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_positivesGdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
validate_shape(*
_output_shapes	
:╚
э
:dnn/head/metrics/auc_precision_recall/false_positives/readIdentity5dnn/head/metrics/auc_precision_recall/false_positives*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:╚
═
2dnn/head/metrics/auc_precision_recall/LogicalAnd_3
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_1-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:╚         
н
/dnn/head/metrics/auc_precision_recall/ToFloat_5Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_3*

SrcT0
*(
_output_shapes
:╚         *

DstT0
─
+dnn/head/metrics/auc_precision_recall/mul_3Mul/dnn/head/metrics/auc_precision_recall/ToFloat_5,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         

=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
+dnn/head/metrics/auc_precision_recall/Sum_3Sum+dnn/head/metrics/auc_precision_recall/mul_3=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
е
1dnn/head/metrics/auc_precision_recall/AssignAdd_3	AssignAdd5dnn/head/metrics/auc_precision_recall/false_positives+dnn/head/metrics/auc_precision_recall/Sum_3*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:╚*
use_locking( 
p
+dnn/head/metrics/auc_precision_recall/add/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
╛
)dnn/head/metrics/auc_precision_recall/addAdd9dnn/head/metrics/auc_precision_recall/true_positives/read+dnn/head/metrics/auc_precision_recall/add/y*
T0*
_output_shapes	
:╚
╧
+dnn/head/metrics/auc_precision_recall/add_1Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_negatives/read*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_2/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┤
+dnn/head/metrics/auc_precision_recall/add_2Add+dnn/head/metrics/auc_precision_recall/add_1-dnn/head/metrics/auc_precision_recall/add_2/y*
T0*
_output_shapes	
:╚
▓
)dnn/head/metrics/auc_precision_recall/divRealDiv)dnn/head/metrics/auc_precision_recall/add+dnn/head/metrics/auc_precision_recall/add_2*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_3/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┬
+dnn/head/metrics/auc_precision_recall/add_3Add9dnn/head/metrics/auc_precision_recall/true_positives/read-dnn/head/metrics/auc_precision_recall/add_3/y*
T0*
_output_shapes	
:╚
╧
+dnn/head/metrics/auc_precision_recall/add_4Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_positives/read*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_5/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┤
+dnn/head/metrics/auc_precision_recall/add_5Add+dnn/head/metrics/auc_precision_recall/add_4-dnn/head/metrics/auc_precision_recall/add_5/y*
T0*
_output_shapes	
:╚
╢
+dnn/head/metrics/auc_precision_recall/div_1RealDiv+dnn/head/metrics/auc_precision_recall/add_3+dnn/head/metrics/auc_precision_recall/add_5*
T0*
_output_shapes	
:╚
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┬
5dnn/head/metrics/auc_precision_recall/strided_slice_1StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_1/stack=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
Index0*
T0*
shrink_axis_mask 
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┬
5dnn/head/metrics/auc_precision_recall/strided_slice_2StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_2/stack=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
─
)dnn/head/metrics/auc_precision_recall/subSub5dnn/head/metrics/auc_precision_recall/strided_slice_15dnn/head/metrics/auc_precision_recall/strided_slice_2*
T0*
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_3StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_3/stack=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
T0*
Index0
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_4StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_4/stack=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟*
T0*
Index0
╞
+dnn/head/metrics/auc_precision_recall/add_6Add5dnn/head/metrics/auc_precision_recall/strided_slice_35dnn/head/metrics/auc_precision_recall/strided_slice_4*
T0*
_output_shapes	
:╟
t
/dnn/head/metrics/auc_precision_recall/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
╝
-dnn/head/metrics/auc_precision_recall/truedivRealDiv+dnn/head/metrics/auc_precision_recall/add_6/dnn/head/metrics/auc_precision_recall/truediv/y*
T0*
_output_shapes	
:╟
░
)dnn/head/metrics/auc_precision_recall/MulMul)dnn/head/metrics/auc_precision_recall/sub-dnn/head/metrics/auc_precision_recall/truediv*
T0*
_output_shapes	
:╟
w
-dnn/head/metrics/auc_precision_recall/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
╩
+dnn/head/metrics/auc_precision_recall/valueSum)dnn/head/metrics/auc_precision_recall/Mul-dnn/head/metrics/auc_precision_recall/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
r
-dnn/head/metrics/auc_precision_recall/add_7/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
╕
+dnn/head/metrics/auc_precision_recall/add_7Add/dnn/head/metrics/auc_precision_recall/AssignAdd-dnn/head/metrics/auc_precision_recall/add_7/y*
T0*
_output_shapes	
:╚
╝
+dnn/head/metrics/auc_precision_recall/add_8Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_1*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_9/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┤
+dnn/head/metrics/auc_precision_recall/add_9Add+dnn/head/metrics/auc_precision_recall/add_8-dnn/head/metrics/auc_precision_recall/add_9/y*
T0*
_output_shapes	
:╚
╢
+dnn/head/metrics/auc_precision_recall/div_2RealDiv+dnn/head/metrics/auc_precision_recall/add_7+dnn/head/metrics/auc_precision_recall/add_9*
T0*
_output_shapes	
:╚
s
.dnn/head/metrics/auc_precision_recall/add_10/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
║
,dnn/head/metrics/auc_precision_recall/add_10Add/dnn/head/metrics/auc_precision_recall/AssignAdd.dnn/head/metrics/auc_precision_recall/add_10/y*
T0*
_output_shapes	
:╚
╜
,dnn/head/metrics/auc_precision_recall/add_11Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_3*
T0*
_output_shapes	
:╚
s
.dnn/head/metrics/auc_precision_recall/add_12/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
╖
,dnn/head/metrics/auc_precision_recall/add_12Add,dnn/head/metrics/auc_precision_recall/add_11.dnn/head/metrics/auc_precision_recall/add_12/y*
T0*
_output_shapes	
:╚
╕
+dnn/head/metrics/auc_precision_recall/div_3RealDiv,dnn/head/metrics/auc_precision_recall/add_10,dnn/head/metrics/auc_precision_recall/add_12*
T0*
_output_shapes	
:╚
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_5StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_5/stack=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_6StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_6/stack=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
╞
+dnn/head/metrics/auc_precision_recall/sub_1Sub5dnn/head/metrics/auc_precision_recall/strided_slice_55dnn/head/metrics/auc_precision_recall/strided_slice_6*
T0*
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_7StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_7/stack=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_8StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_8/stack=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
╟
,dnn/head/metrics/auc_precision_recall/add_13Add5dnn/head/metrics/auc_precision_recall/strided_slice_75dnn/head/metrics/auc_precision_recall/strided_slice_8*
T0*
_output_shapes	
:╟
v
1dnn/head/metrics/auc_precision_recall/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
┴
/dnn/head/metrics/auc_precision_recall/truediv_1RealDiv,dnn/head/metrics/auc_precision_recall/add_131dnn/head/metrics/auc_precision_recall/truediv_1/y*
T0*
_output_shapes	
:╟
╢
+dnn/head/metrics/auc_precision_recall/Mul_1Mul+dnn/head/metrics/auc_precision_recall/sub_1/dnn/head/metrics/auc_precision_recall/truediv_1*
T0*
_output_shapes	
:╟
w
-dnn/head/metrics/auc_precision_recall/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
╨
/dnn/head/metrics/auc_precision_recall/update_opSum+dnn/head/metrics/auc_precision_recall/Mul_1-dnn/head/metrics/auc_precision_recall/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
А
mean/total/Initializer/zerosConst*
_class
loc:@mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
Н

mean/total
VariableV2*
shared_name *
_class
loc:@mean/total*
	container *
shape: *
dtype0*
_output_shapes
: 
о
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: 
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 
А
mean/count/Initializer/zerosConst*
_class
loc:@mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
Н

mean/count
VariableV2*
shared_name *
_class
loc:@mean/count*
	container *
shape: *
dtype0*
_output_shapes
: 
о
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
T0*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: *
use_locking(
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*
_output_shapes
: *

DstT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
u
mean/SumSumdnn/head/weighted_loss/Sum
mean/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Д
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
T0*
_class
loc:@mean/total*
_output_shapes
: *
use_locking( 
й
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^dnn/head/weighted_loss/Sum*
use_locking( *
T0*
_class
loc:@mean/count*
_output_shapes
: 
S
mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
mean/GreaterGreatermean/count/readmean/Greater/y*
T0*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
Q
mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_

mean/valueSelectmean/Greatermean/truedivmean/value/e*
T0*
_output_shapes
: 
U
mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
mean/Greater_1Greatermean/AssignAdd_1mean/Greater_1/y*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
U
mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
mean/update_opSelectmean/Greater_1mean/truediv_1mean/update_op/e*
T0*
_output_shapes
: 
╩

group_depsNoOp$^dnn/head/metrics/accuracy/update_op-^dnn/head/metrics/accuracy_baseline/update_op^dnn/head/metrics/auc/update_op0^dnn/head/metrics/auc_precision_recall/update_op(^dnn/head/metrics/average_loss/update_op&^dnn/head/metrics/label/mean/update_op^mean/update_op+^dnn/head/metrics/prediction/mean/update_op
{
eval_step/Initializer/zerosConst*
_class
loc:@eval_step*
value	B	 R *
dtype0	*
_output_shapes
: 
Л
	eval_step
VariableV2*
shared_name *
_class
loc:@eval_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
к
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: *
use_locking(
d
eval_step/readIdentity	eval_step*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking(*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
U
readIdentity	eval_step^group_deps
^AssignAdd*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
T0	*
_output_shapes
: 
С
initNoOp^global_step/Assign`^dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/AssignZ^dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Assign'^dnn/hiddenlayer_0/kernel/part_0/Assign%^dnn/hiddenlayer_0/bias/part_0/Assign'^dnn/hiddenlayer_1/kernel/part_0/Assign%^dnn/hiddenlayer_1/bias/part_0/Assign'^dnn/hiddenlayer_2/kernel/part_0/Assign%^dnn/hiddenlayer_2/bias/part_0/Assign ^dnn/logits/kernel/part_0/Assign^dnn/logits/bias/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
п
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
┼
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
┼
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
┼
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_2/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
╕
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
╬
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized!dnn/head/metrics/label/mean/total*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
dtype0*
_output_shapes
: 
╬
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized!dnn/head/metrics/label/mean/count*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized#dnn/head/metrics/average_loss/total*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
╩
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializeddnn/head/metrics/accuracy/total*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
╩
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializeddnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
╪
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized&dnn/head/metrics/prediction/mean/total*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
dtype0*
_output_shapes
: 
╪
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized&dnn/head/metrics/prediction/mean/count*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized#dnn/head/metrics/auc/true_positives*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
dtype0*
_output_shapes
: 
╘
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized$dnn/head/metrics/auc/false_negatives*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized#dnn/head/metrics/auc/true_negatives*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
dtype0*
_output_shapes
: 
╘
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitialized$dnn/head/metrics/auc/false_positives*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
dtype0*
_output_shapes
: 
Ї
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_positives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
dtype0*
_output_shapes
: 
Ў
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_negatives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
dtype0*
_output_shapes
: 
Ї
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_negatives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
dtype0*
_output_shapes
: 
Ў
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_positives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
Ч
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_29"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
И
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
▀	
$report_uninitialized_variables/ConstConst"/device:CPU:0*ў
valueэBъBglobal_stepBXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0BRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0Bdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/hiddenlayer_2/kernel/part_0Bdnn/hiddenlayer_2/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0B!dnn/head/metrics/label/mean/totalB!dnn/head/metrics/label/mean/countB#dnn/head/metrics/average_loss/totalB#dnn/head/metrics/average_loss/countBdnn/head/metrics/accuracy/totalBdnn/head/metrics/accuracy/countB&dnn/head/metrics/prediction/mean/totalB&dnn/head/metrics/prediction/mean/countB#dnn/head/metrics/auc/true_positivesB$dnn/head/metrics/auc/false_negativesB#dnn/head/metrics/auc/true_negativesB$dnn/head/metrics/auc/false_positivesB4dnn/head/metrics/auc_precision_recall/true_positivesB5dnn/head/metrics/auc_precision_recall/false_negativesB4dnn/head/metrics/auc_precision_recall/true_negativesB5dnn/head/metrics/auc_precision_recall/false_positivesB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
К
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ш
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Ы
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Д
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
М
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ё
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
╛
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
И
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
║
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
┌
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
Э
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
         *
dtype0*
_output_shapes
:
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
й
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:         
┼
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
T0	*#
_output_shapes
:         *
squeeze_dims

С
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:         
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*

Tidx0*
T0*
N*#
_output_shapes
:         
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
▒
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
╟
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
╟
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
╟
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_2/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
║
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
Ї
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_10"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
М
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
м
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*┬
value╕B╡Bglobal_stepBXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0BRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0Bdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/hiddenlayer_2/kernel/part_0Bdnn/hiddenlayer_2/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
М
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Є
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Э
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
К
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
О
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ю
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ю
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
·
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0
┬
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
К
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
┬
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
Я
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
         *
dtype0*
_output_shapes
:
Ё
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
н
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:         
╔
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
T0	*#
_output_shapes
:         *
squeeze_dims

Ч
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:         
█
init_2NoOp)^dnn/head/metrics/label/mean/total/Assign)^dnn/head/metrics/label/mean/count/Assign+^dnn/head/metrics/average_loss/total/Assign+^dnn/head/metrics/average_loss/count/Assign'^dnn/head/metrics/accuracy/total/Assign'^dnn/head/metrics/accuracy/count/Assign.^dnn/head/metrics/prediction/mean/total/Assign.^dnn/head/metrics/prediction/mean/count/Assign+^dnn/head/metrics/auc/true_positives/Assign,^dnn/head/metrics/auc/false_negatives/Assign+^dnn/head/metrics/auc/true_negatives/Assign,^dnn/head/metrics/auc/false_positives/Assign<^dnn/head/metrics/auc_precision_recall/true_positives/Assign=^dnn/head/metrics/auc_precision_recall/false_negatives/Assign<^dnn/head/metrics/auc_precision_recall/true_negatives/Assign=^dnn/head/metrics/auc_precision_recall/false_positives/Assign^mean/total/Assign^mean/count/Assign^eval_step/Assign
▀
init_all_tablesNoOph^dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init\^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
∙
Merge/MergeSummaryMergeSummary-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation-dnn/dnn/hiddenlayer_2/fraction_of_zero_values dnn/dnn/hiddenlayer_2/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_37663ef8eafb4953b0fd37844185b92f/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
╔
save/SaveV2/tensor_namesConst*№
valueЄBяBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBQdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weightsBKdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weightsBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
Ї
save/SaveV2/shape_and_slicesConst*г
valueЩBЦB50 0,50B57 50 0,57:0,50B	100 0,100B50 100 0,50:0,100B50 0,50B100 50 0,100:0,50B12 12 0,12:0,12B32 32 0,32:0,32B1 0,1B50 1 0,50:0,1B *
dtype0*
_output_shapes
:
▌
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/read]dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/readWdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
z
save/RestoreV2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:
o
save/RestoreV2/shape_and_slicesConst*
valueBB50 0,50*
dtype0*
_output_shapes
:
Т
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:2
─
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:2
~
save/RestoreV2_1/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
:
y
!save/RestoreV2_1/shape_and_slicesConst*$
valueBB57 50 0,57:0,50*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes

:92
╨
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2_1*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:92*
use_locking(
|
save/RestoreV2_2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_1/bias*
dtype0*
_output_shapes
:
s
!save/RestoreV2_2/shape_and_slicesConst*
valueBB	100 0,100*
dtype0*
_output_shapes
:
Ш
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:d
╚
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2_2*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:d
~
save/RestoreV2_3/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
:
{
!save/RestoreV2_3/shape_and_slicesConst*&
valueBB50 100 0,50:0,100*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes

:2d
╨
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2_3*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:2d
|
save/RestoreV2_4/tensor_namesConst*+
value"B Bdnn/hiddenlayer_2/bias*
dtype0*
_output_shapes
:
q
!save/RestoreV2_4/shape_and_slicesConst*
valueBB50 0,50*
dtype0*
_output_shapes
:
Ш
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:2*
dtypes
2
╚
save/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save/RestoreV2_4*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes
:2
~
save/RestoreV2_5/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/kernel*
dtype0*
_output_shapes
:
{
!save/RestoreV2_5/shape_and_slicesConst*&
valueBB100 50 0,100:0,50*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes

:d2*
dtypes
2
╨
save/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save/RestoreV2_5*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(*
_output_shapes

:d2*
use_locking(
╖
save/RestoreV2_6/tensor_namesConst*f
value]B[BQdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights*
dtype0*
_output_shapes
:
y
!save/RestoreV2_6/shape_and_slicesConst*$
valueBB12 12 0,12:0,12*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes

:
┬
save/Assign_6AssignXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0save/RestoreV2_6*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:*
use_locking(
▒
save/RestoreV2_7/tensor_namesConst*`
valueWBUBKdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights*
dtype0*
_output_shapes
:
y
!save/RestoreV2_7/shape_and_slicesConst*$
valueBB32 32 0,32:0,32*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes

:  
╢
save/Assign_7AssignRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0save/RestoreV2_7*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:  *
use_locking(
u
save/RestoreV2_8/tensor_namesConst*$
valueBBdnn/logits/bias*
dtype0*
_output_shapes
:
o
!save/RestoreV2_8/shape_and_slicesConst*
valueBB1 0,1*
dtype0*
_output_shapes
:
Ш
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
║
save/Assign_8Assigndnn/logits/bias/part_0save/RestoreV2_8*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(
w
save/RestoreV2_9/tensor_namesConst*&
valueBBdnn/logits/kernel*
dtype0*
_output_shapes
:
w
!save/RestoreV2_9/shape_and_slicesConst*"
valueBB50 1 0,50:0,1*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes

:2
┬
save/Assign_9Assigndnn/logits/kernel/part_0save/RestoreV2_9*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:2
r
save/RestoreV2_10/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2	
в
save/Assign_10Assignglobal_stepsave/RestoreV2_10*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
╔
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10
-
save/restore_allNoOp^save/restore_shardЎ
є
*
_make_dataset_ad3a09d5
BatchDataset{
&TensorSliceDataset/tensors/component_0Const*=
value4B2
"(l  S  О  K  W  к  ;  ▌  p    *
dtype0{
&TensorSliceDataset/tensors/component_1Const*=
value4B2
"(Й  =    2  └   $  C  H  █  э  *
dtype0╧
&TensorSliceDataset/tensors/component_2Const*Р
valueЖBГ
BSoutheasternBSoutheasternBBig TenBBig 12BAtlantic CoastBAtlantic CoastBPac-12BMountain WestBBig TenBSoutheastern*
dtype0{
&TensorSliceDataset/tensors/component_3Const*=
value4B2
"(▌  ╘  ╤  ╤  ╤  ╧  ╧  ┘  ┘  ╫  *
dtype0{
&TensorSliceDataset/tensors/component_4Const*=
value4B2
"(/   +   -         -      -   .   $   *
dtype0{
&TensorSliceDataset/tensors/component_5Const*=
value4B2
"(                                        *
dtype0{
&TensorSliceDataset/tensors/component_6Const*=
value4B2
"('   #   -                   .      *
dtype0{
&TensorSliceDataset/tensors/component_7Const*=
value4B2
"(э          m            о   ╔      *
dtype0{
&TensorSliceDataset/tensors/component_8Const*=
value4B2
"(                              *
dtype0{
&TensorSliceDataset/tensors/component_9Const*=
value4B2
"(  А   №   д   ы   ╤  л   Д   с   Л   *
dtype0|
'TensorSliceDataset/tensors/component_10Const*=
value4B2
"(                              *
dtype0|
'TensorSliceDataset/tensors/component_11Const*=
value4B2
"(  y   Д  Ф      o  Ш   \  O   *
dtype0Д
'TensorSliceDataset/tensors/component_12Const*E
value<B:
BSFOBSDGBSDGBWASBATLBPHIBCINBDENBINDBOAK*
dtype0|
'TensorSliceDataset/tensors/component_13Const*=
value4B2
"(4   Q   Z         M   -   /   C   4   *
dtype0|
'TensorSliceDataset/tensors/component_14Const*=
value4B2
"(ё   З'  .  D  у  ┼     ╔  Ы+  с  *
dtype0|
'TensorSliceDataset/tensors/component_15Const*=
value4B2
"(                                    *
dtype0Ъ
TensorSliceDatasetTensorSliceDataset/TensorSliceDataset/tensors/component_0:output:0/TensorSliceDataset/tensors/component_1:output:0/TensorSliceDataset/tensors/component_2:output:0/TensorSliceDataset/tensors/component_3:output:0/TensorSliceDataset/tensors/component_4:output:0/TensorSliceDataset/tensors/component_5:output:0/TensorSliceDataset/tensors/component_6:output:0/TensorSliceDataset/tensors/component_7:output:0/TensorSliceDataset/tensors/component_8:output:0/TensorSliceDataset/tensors/component_9:output:00TensorSliceDataset/tensors/component_10:output:00TensorSliceDataset/tensors/component_11:output:00TensorSliceDataset/tensors/component_12:output:00TensorSliceDataset/tensors/component_13:output:00TensorSliceDataset/tensors/component_14:output:00TensorSliceDataset/tensors/component_15:output:0*3
output_shapes"
 : : : : : : : : : : : : : : : : *%
Toutput_types
2A
BatchDataset/batch_sizeConst*
value	B	 R
*
dtype0	Й
BatchDatasetBatchDatasetTensorSliceDataset:handle:0 BatchDataset/batch_size:output:0*$
output_types
2*Е
output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         "%
BatchDatasetBatchDataset:handle:0"HQ3В     ~┬(	╣n#HОК╓AJїЬ
Ф8Є7
9
Add
x"T
y"T
z"T"
Ttype:
2	
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
Ш
ArgMax

input"T
	dimension"Tidx
output"output_type"
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
╢
AsString

input"T

output"
Ttype:
	2	
"
	precisionint         "

scientificbool( "
shortestbool( "
widthint         "
fillstring 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
p
	AssignAdd
ref"TА

value"T

output_ref"TА"
Ttype:
2	"
use_lockingbool( 
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
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
╣
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
+
Exp
x"T
y"T"
Ttype:	
2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
М
Gather
params"Tparams
indices"Tindices
output"Tparams"
validate_indicesbool("
Tparamstype"
Tindicestype:
2	
:
Greater
x"T
y"T
z
"
Ttype:
2		
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
б
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
╔
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0■        "
value_indexint(0■        "+

vocab_sizeint         (0         "
	delimiterstring	И
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ
А
IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0И
7
Less
x"T
y"T
z
"
Ttype:
2		
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
-
Log1p
x"T
y"T"
Ttype:	
2
$

LogicalAnd
x

y

z
Р


LogicalNot
x

y

w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	Р
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	
Р
п
OneShotIterator

handle"
dataset_factoryfunc"
output_types
list(type)(0" 
output_shapeslist(shape)(0"
	containerstring "
shared_namestring И
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
)
Rank

input"T

output"	
Ttype
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
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
/
Sigmoid
x"T
y"T"
Ttype:	
2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
╖
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ў
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
9
Sub
x"T
y"T
z"T"
Ttype:
2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И

Where	
input
	
index	
&
	ZerosLike
x"T
y"T"	
Ttype*1.4.02v1.4.0-rc1-11-g130a514┼Ф

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
П
global_step
VariableV2*
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: *
shared_name 
▓
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
У
tensors/component_0Const"/device:CPU:0*=
value4B2
"(l  S  О  K  W  к  ;  ▌  p    *
dtype0*
_output_shapes
:

У
tensors/component_1Const"/device:CPU:0*=
value4B2
"(Й  =    2  └   $  C  H  █  э  *
dtype0*
_output_shapes
:

ч
tensors/component_2Const"/device:CPU:0*Р
valueЖBГ
BSoutheasternBSoutheasternBBig TenBBig 12BAtlantic CoastBAtlantic CoastBPac-12BMountain WestBBig TenBSoutheastern*
dtype0*
_output_shapes
:

У
tensors/component_3Const"/device:CPU:0*=
value4B2
"(▌  ╘  ╤  ╤  ╤  ╧  ╧  ┘  ┘  ╫  *
dtype0*
_output_shapes
:

У
tensors/component_4Const"/device:CPU:0*=
value4B2
"(/   +   -         -      -   .   $   *
dtype0*
_output_shapes
:

У
tensors/component_5Const"/device:CPU:0*=
value4B2
"(                                        *
dtype0*
_output_shapes
:

У
tensors/component_6Const"/device:CPU:0*=
value4B2
"('   #   -                   .      *
dtype0*
_output_shapes
:

У
tensors/component_7Const"/device:CPU:0*=
value4B2
"(э          m            о   ╔      *
dtype0*
_output_shapes
:

У
tensors/component_8Const"/device:CPU:0*=
value4B2
"(                              *
dtype0*
_output_shapes
:

У
tensors/component_9Const"/device:CPU:0*=
value4B2
"(  А   №   д   ы   ╤  л   Д   с   Л   *
dtype0*
_output_shapes
:

Ф
tensors/component_10Const"/device:CPU:0*=
value4B2
"(                              *
dtype0*
_output_shapes
:

Ф
tensors/component_11Const"/device:CPU:0*=
value4B2
"(  y   Д  Ф      o  Ш   \  O   *
dtype0*
_output_shapes
:

Ь
tensors/component_12Const"/device:CPU:0*E
value<B:
BSFOBSDGBSDGBWASBATLBPHIBCINBDENBINDBOAK*
dtype0*
_output_shapes
:

Ф
tensors/component_13Const"/device:CPU:0*=
value4B2
"(4   Q   Z         M   -   /   C   4   *
dtype0*
_output_shapes
:

Ф
tensors/component_14Const"/device:CPU:0*=
value4B2
"(ё   З'  .  D  у  ┼     ╔  Ы+  с  *
dtype0*
_output_shapes
:

Ф
tensors/component_15Const"/device:CPU:0*=
value4B2
"(                                    *
dtype0*
_output_shapes
:

╩
OneShotIteratorOneShotIterator"/device:CPU:0*$
output_types
2*
_output_shapes
: *-
dataset_factoryR
_make_dataset_ad3a09d5*
shared_name *Е
output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         *
	container 
∙
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*$
output_types
2*Е
output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         *Ж
_output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         
Н
Bdnn/input_from_feature_columns/input_layer/Attempts/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
▀
>dnn/input_from_feature_columns/input_layer/Attempts/ExpandDims
ExpandDimsIteratorGetNextBdnn/input_from_feature_columns/input_layer/Attempts/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
─
;dnn/input_from_feature_columns/input_layer/Attempts/ToFloatCast>dnn/input_from_feature_columns/input_layer/Attempts/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
┤
9dnn/input_from_feature_columns/input_layer/Attempts/ShapeShape;dnn/input_from_feature_columns/input_layer/Attempts/ToFloat*
T0*
out_type0*
_output_shapes
:
С
Gdnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¤
Adnn/input_from_feature_columns/input_layer/Attempts/strided_sliceStridedSlice9dnn/input_from_feature_columns/input_layer/Attempts/ShapeGdnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stackIdnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_1Idnn/input_from_feature_columns/input_layer/Attempts/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Е
Cdnn/input_from_feature_columns/input_layer/Attempts/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Л
Adnn/input_from_feature_columns/input_layer/Attempts/Reshape/shapePackAdnn/input_from_feature_columns/input_layer/Attempts/strided_sliceCdnn/input_from_feature_columns/input_layer/Attempts/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Ж
;dnn/input_from_feature_columns/input_layer/Attempts/ReshapeReshape;dnn/input_from_feature_columns/input_layer/Attempts/ToFloatAdnn/input_from_feature_columns/input_layer/Attempts/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Р
Ednn/input_from_feature_columns/input_layer/Completions/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ч
Adnn/input_from_feature_columns/input_layer/Completions/ExpandDims
ExpandDimsIteratorGetNext:1Ednn/input_from_feature_columns/input_layer/Completions/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╩
>dnn/input_from_feature_columns/input_layer/Completions/ToFloatCastAdnn/input_from_feature_columns/input_layer/Completions/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
║
<dnn/input_from_feature_columns/input_layer/Completions/ShapeShape>dnn/input_from_feature_columns/input_layer/Completions/ToFloat*
T0*
out_type0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/Completions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
М
Ddnn/input_from_feature_columns/input_layer/Completions/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/Completions/ShapeJdnn/input_from_feature_columns/input_layer/Completions/strided_slice/stackLdnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/Completions/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
И
Fdnn/input_from_feature_columns/input_layer/Completions/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ф
Ddnn/input_from_feature_columns/input_layer/Completions/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/Completions/strided_sliceFdnn/input_from_feature_columns/input_layer/Completions/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
П
>dnn/input_from_feature_columns/input_layer/Completions/ReshapeReshape>dnn/input_from_feature_columns/input_layer/Completions/ToFloatDdnn/input_from_feature_columns/input_layer/Completions/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Щ
Ndnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
∙
Jdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims
ExpandDimsIteratorGetNext:2Ndnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
▀
Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/ShapeShapeJdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims*
T0*
out_type0*
_output_shapes
:
ч
Tdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/CastCastUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Shape*

SrcT0*
_output_shapes
:*

DstT0	
Щ
Xdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast_1/xConst*
valueB B *
dtype0*
_output_shapes
: 
╝
Xdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/NotEqualNotEqualJdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDimsXdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
с
Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/WhereWhereXdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         
░
]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╔
Wdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/ReshapeReshapeJdnn/input_from_feature_columns/input_layer/Conference_embedding/ExpandDims]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
┤
cdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stackConst*
valueB"       *
dtype0*
_output_shapes
:
╢
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╢
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Ц
]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_sliceStridedSliceUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Wherecdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stackednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_1ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:         
╢
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
╕
gdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╕
gdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
в
_dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1StridedSliceUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Whereednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stackgdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_1gdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:         *
T0	*
Index0
ё
Wdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/unstackUnpackTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast*
T0	*	
num*

axis *
_output_shapes
: : 
Є
Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/stackPackYdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/unstack:1*
T0	*

axis *
N*
_output_shapes
:
─
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/MulMul_dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_slice_1Udnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/stack*
T0	*'
_output_shapes
:         
п
ednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
с
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/SumSumSdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Mulednn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Sum/reduction_indices*
T0	*#
_output_shapes
:         *

Tidx0*
	keep_dims( 
╝
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/AddAdd]dnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/strided_sliceSdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
ъ
Vdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/GatherGatherWdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/ReshapeSdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Add*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:         
Х
\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_tableHashTableV2*>
shared_name/-hash_table_vocab_list/conference.txt_12_-2_-1*
use_node_name_sharing( *
	key_dtype0*
	container *
value_dtype0	*
_output_shapes
: 
н
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/ConstConst*
valueB	 R
         *
dtype0	*
_output_shapes
: 
╨
vdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init/asset_filepathConst**
value!B Bvocab_list/conference.txt*
dtype0*
_output_shapes
: 
╕
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_initInitializeTableFromTextFileV2\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_tablevdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init/asset_filepath*
	delimiter	*
	key_index■        *
value_index         *

vocab_size
╝
Qdnn/input_from_feature_columns/input_layer/Conference_embedding/hash_table_LookupLookupTableFindV2\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_tableVdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Gatherbdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:         *	
Tin0
╣
{dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
valueB"      *
dtype0*
_output_shapes
:
м
zdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
о
|dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
valueB
 *:═У>*
dtype0*
_output_shapes
: 
╥
Еdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes

:*

seed *
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
seed2 
Ь
ydnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMulЕdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
_output_shapes

:
Й
udnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
_output_shapes

:
╣
Xdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0
VariableV2*
shared_name *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
	container *
shape
:*
dtype0*
_output_shapes

:
∙
_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:*
use_locking(
┘
]dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
_output_shapes

:
▓
hdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
▒
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╞
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SliceSliceTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Casthdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
м
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
я
adnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Const*

Tidx0*
	keep_dims( *
T0	*
_output_shapes
: 
н
kdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
 
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GatherGatherTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Castkdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather/indices*
Tparams0	*
validate_indices(*
_output_shapes
: *
Tindices0
э
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather*
T0	*

axis *
N*
_output_shapes
:
╝
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshapeSparseReshapeUdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/WhereTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Castcdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast/x*-
_output_shapes
:         :
А
sdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape/IdentityIdentityQdnn/input_from_feature_columns/input_layer/Conference_embedding/hash_table_Lookup*
T0	*#
_output_shapes
:         
н
kdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
Й
idnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
 
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/GreaterEqual*'
_output_shapes
:         
╜
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
√
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:         
б
ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_1Gatherjdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape*
Tindices0	*
Tparams0	*
validate_indices(*'
_output_shapes
:         
ж
ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_2Gathersdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape*
Tparams0	*
validate_indices(*#
_output_shapes
:         *
Tindices0	
Д
ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
╕
vdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
и
Дdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_1ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Gather_2ednn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
┌
Иdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
▄
Кdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
▄
Кdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
▐
Вdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceДdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsИdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stackКdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Кdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*#
_output_shapes
:         *
T0	*
Index0
├
ydnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/CastCastВdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:         *

DstT0
┌
{dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/UniqueUniqueЖdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*
out_idx0*2
_output_shapes 
:         :         
╣
Еdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/embedding_lookupGather]dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/Unique*
Tparams0*
validate_indices(*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*'
_output_shapes
:         *
Tindices0	
╔
tdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparseSparseSegmentMeanЕdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/embedding_lookup}dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:         *

Tidx0
╜
ldnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
и
fdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1ReshapeЖdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
Ц
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
║
pdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
╝
rdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
╝
rdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╩
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
ж
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
Ў
bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/strided_slice*
T0*

axis *
N*
_output_shapes
:
В
adnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
Ь
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
┌
\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:         
ї
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast_1CastTdnn/input_from_feature_columns/input_layer/Conference_embedding/to_sparse_input/Cast*

SrcT0	*
_output_shapes
:*

DstT0
┤
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
│
idnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
█
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
А
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights*
T0*
out_type0*
_output_shapes
:
┤
jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
╝
idnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
▄
ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
к
hdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
▀
cdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ї
fdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weightscdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:         
ы
Ednn/input_from_feature_columns/input_layer/Conference_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
Э
Sdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Я
Udnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Я
Udnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╣
Mdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/Conference_embedding/ShapeSdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/Conference_embedding/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
С
Odnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
п
Mdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/Conference_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
╔
Gdnn/input_from_feature_columns/input_layer/Conference_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
О
Cdnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
у
?dnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims
ExpandDimsIteratorGetNext:3Cdnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╞
<dnn/input_from_feature_columns/input_layer/DraftYear/ToFloatCast?dnn/input_from_feature_columns/input_layer/DraftYear/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╢
:dnn/input_from_feature_columns/input_layer/DraftYear/ShapeShape<dnn/input_from_feature_columns/input_layer/DraftYear/ToFloat*
T0*
out_type0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
В
Bdnn/input_from_feature_columns/input_layer/DraftYear/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/DraftYear/ShapeHdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stackJdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/DraftYear/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Ж
Ddnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
О
Bdnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/DraftYear/strided_sliceDdnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Й
<dnn/input_from_feature_columns/input_layer/DraftYear/ReshapeReshape<dnn/input_from_feature_columns/input_layer/DraftYear/ToFloatBdnn/input_from_feature_columns/input_layer/DraftYear/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Р
Ednn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ч
Adnn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims
ExpandDimsIteratorGetNext:4Ednn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╩
>dnn/input_from_feature_columns/input_layer/GamesPlayed/ToFloatCastAdnn/input_from_feature_columns/input_layer/GamesPlayed/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
║
<dnn/input_from_feature_columns/input_layer/GamesPlayed/ShapeShape>dnn/input_from_feature_columns/input_layer/GamesPlayed/ToFloat*
T0*
out_type0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
М
Ddnn/input_from_feature_columns/input_layer/GamesPlayed/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/GamesPlayed/ShapeJdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stackLdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/GamesPlayed/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
И
Fdnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ф
Ddnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/GamesPlayed/strided_sliceFdnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
П
>dnn/input_from_feature_columns/input_layer/GamesPlayed/ReshapeReshape>dnn/input_from_feature_columns/input_layer/GamesPlayed/ToFloatDdnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
М
Adnn/input_from_feature_columns/input_layer/Heisman/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
▀
=dnn/input_from_feature_columns/input_layer/Heisman/ExpandDims
ExpandDimsIteratorGetNext:5Adnn/input_from_feature_columns/input_layer/Heisman/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
┬
:dnn/input_from_feature_columns/input_layer/Heisman/ToFloatCast=dnn/input_from_feature_columns/input_layer/Heisman/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
▓
8dnn/input_from_feature_columns/input_layer/Heisman/ShapeShape:dnn/input_from_feature_columns/input_layer/Heisman/ToFloat*
T0*
out_type0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
°
@dnn/input_from_feature_columns/input_layer/Heisman/strided_sliceStridedSlice8dnn/input_from_feature_columns/input_layer/Heisman/ShapeFdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stackHdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_1Hdnn/input_from_feature_columns/input_layer/Heisman/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Д
Bdnn/input_from_feature_columns/input_layer/Heisman/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
И
@dnn/input_from_feature_columns/input_layer/Heisman/Reshape/shapePack@dnn/input_from_feature_columns/input_layer/Heisman/strided_sliceBdnn/input_from_feature_columns/input_layer/Heisman/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Г
:dnn/input_from_feature_columns/input_layer/Heisman/ReshapeReshape:dnn/input_from_feature_columns/input_layer/Heisman/ToFloat@dnn/input_from_feature_columns/input_layer/Heisman/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Т
Gdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ы
Cdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims
ExpandDimsIteratorGetNext:6Gdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╬
@dnn/input_from_feature_columns/input_layer/Interceptions/ToFloatCastCdnn/input_from_feature_columns/input_layer/Interceptions/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╛
>dnn/input_from_feature_columns/input_layer/Interceptions/ShapeShape@dnn/input_from_feature_columns/input_layer/Interceptions/ToFloat*
T0*
out_type0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ц
Fdnn/input_from_feature_columns/input_layer/Interceptions/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/Interceptions/ShapeLdnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stackNdnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/Interceptions/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
К
Hdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ъ
Fdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/Interceptions/strided_sliceHdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Х
@dnn/input_from_feature_columns/input_layer/Interceptions/ReshapeReshape@dnn/input_from_feature_columns/input_layer/Interceptions/ToFloatFdnn/input_from_feature_columns/input_layer/Interceptions/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Й
>dnn/input_from_feature_columns/input_layer/Pick/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
┘
:dnn/input_from_feature_columns/input_layer/Pick/ExpandDims
ExpandDimsIteratorGetNext:7>dnn/input_from_feature_columns/input_layer/Pick/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╝
7dnn/input_from_feature_columns/input_layer/Pick/ToFloatCast:dnn/input_from_feature_columns/input_layer/Pick/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
м
5dnn/input_from_feature_columns/input_layer/Pick/ShapeShape7dnn/input_from_feature_columns/input_layer/Pick/ToFloat*
T0*
out_type0*
_output_shapes
:
Н
Cdnn/input_from_feature_columns/input_layer/Pick/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
П
Ednn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
П
Ednn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
щ
=dnn/input_from_feature_columns/input_layer/Pick/strided_sliceStridedSlice5dnn/input_from_feature_columns/input_layer/Pick/ShapeCdnn/input_from_feature_columns/input_layer/Pick/strided_slice/stackEdnn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_1Ednn/input_from_feature_columns/input_layer/Pick/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Б
?dnn/input_from_feature_columns/input_layer/Pick/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
 
=dnn/input_from_feature_columns/input_layer/Pick/Reshape/shapePack=dnn/input_from_feature_columns/input_layer/Pick/strided_slice?dnn/input_from_feature_columns/input_layer/Pick/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
·
7dnn/input_from_feature_columns/input_layer/Pick/ReshapeReshape7dnn/input_from_feature_columns/input_layer/Pick/ToFloat=dnn/input_from_feature_columns/input_layer/Pick/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
К
?dnn/input_from_feature_columns/input_layer/Round/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
█
;dnn/input_from_feature_columns/input_layer/Round/ExpandDims
ExpandDimsIteratorGetNext:8?dnn/input_from_feature_columns/input_layer/Round/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╛
8dnn/input_from_feature_columns/input_layer/Round/ToFloatCast;dnn/input_from_feature_columns/input_layer/Round/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
о
6dnn/input_from_feature_columns/input_layer/Round/ShapeShape8dnn/input_from_feature_columns/input_layer/Round/ToFloat*
T0*
out_type0*
_output_shapes
:
О
Ddnn/input_from_feature_columns/input_layer/Round/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ю
>dnn/input_from_feature_columns/input_layer/Round/strided_sliceStridedSlice6dnn/input_from_feature_columns/input_layer/Round/ShapeDdnn/input_from_feature_columns/input_layer/Round/strided_slice/stackFdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_1Fdnn/input_from_feature_columns/input_layer/Round/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
В
@dnn/input_from_feature_columns/input_layer/Round/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
В
>dnn/input_from_feature_columns/input_layer/Round/Reshape/shapePack>dnn/input_from_feature_columns/input_layer/Round/strided_slice@dnn/input_from_feature_columns/input_layer/Round/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
¤
8dnn/input_from_feature_columns/input_layer/Round/ReshapeReshape8dnn/input_from_feature_columns/input_layer/Round/ToFloat>dnn/input_from_feature_columns/input_layer/Round/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
С
Fdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
щ
Bdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims
ExpandDimsIteratorGetNext:9Fdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╠
?dnn/input_from_feature_columns/input_layer/RushAttempts/ToFloatCastBdnn/input_from_feature_columns/input_layer/RushAttempts/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╝
=dnn/input_from_feature_columns/input_layer/RushAttempts/ShapeShape?dnn/input_from_feature_columns/input_layer/RushAttempts/ToFloat*
T0*
out_type0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
С
Ednn/input_from_feature_columns/input_layer/RushAttempts/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/RushAttempts/ShapeKdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stackMdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/RushAttempts/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Й
Gdnn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ч
Ednn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/RushAttempts/strided_sliceGdnn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Т
?dnn/input_from_feature_columns/input_layer/RushAttempts/ReshapeReshape?dnn/input_from_feature_columns/input_layer/RushAttempts/ToFloatEdnn/input_from_feature_columns/input_layer/RushAttempts/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
У
Hdnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ю
Ddnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims
ExpandDimsIteratorGetNext:10Hdnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╨
Adnn/input_from_feature_columns/input_layer/RushTouchdowns/ToFloatCastDdnn/input_from_feature_columns/input_layer/RushTouchdowns/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
└
?dnn/input_from_feature_columns/input_layer/RushTouchdowns/ShapeShapeAdnn/input_from_feature_columns/input_layer/RushTouchdowns/ToFloat*
T0*
out_type0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ы
Gdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/RushTouchdowns/ShapeMdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stackOdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Л
Idnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Э
Gdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/RushTouchdowns/strided_sliceIdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Ш
Adnn/input_from_feature_columns/input_layer/RushTouchdowns/ReshapeReshapeAdnn/input_from_feature_columns/input_layer/RushTouchdowns/ToFloatGdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
О
Cdnn/input_from_feature_columns/input_layer/RushYards/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ф
?dnn/input_from_feature_columns/input_layer/RushYards/ExpandDims
ExpandDimsIteratorGetNext:11Cdnn/input_from_feature_columns/input_layer/RushYards/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╞
<dnn/input_from_feature_columns/input_layer/RushYards/ToFloatCast?dnn/input_from_feature_columns/input_layer/RushYards/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╢
:dnn/input_from_feature_columns/input_layer/RushYards/ShapeShape<dnn/input_from_feature_columns/input_layer/RushYards/ToFloat*
T0*
out_type0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
В
Bdnn/input_from_feature_columns/input_layer/RushYards/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/RushYards/ShapeHdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stackJdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/RushYards/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
Ж
Ddnn/input_from_feature_columns/input_layer/RushYards/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
О
Bdnn/input_from_feature_columns/input_layer/RushYards/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/RushYards/strided_sliceDdnn/input_from_feature_columns/input_layer/RushYards/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
Й
<dnn/input_from_feature_columns/input_layer/RushYards/ReshapeReshape<dnn/input_from_feature_columns/input_layer/RushYards/ToFloatBdnn/input_from_feature_columns/input_layer/RushYards/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
У
Hdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ю
Ddnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims
ExpandDimsIteratorGetNext:12Hdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╙
Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/ShapeShapeDdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDims*
T0*
out_type0*
_output_shapes
:
█
Ndnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/CastCastOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Shape*

SrcT0*
_output_shapes
:*

DstT0	
У
Rdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast_1/xConst*
valueB B *
dtype0*
_output_shapes
: 
к
Rdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/NotEqualNotEqualDdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDimsRdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast_1/x*
T0*'
_output_shapes
:         
╒
Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/WhereWhereRdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/NotEqual*'
_output_shapes
:         
к
Wdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╖
Qdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/ReshapeReshapeDdnn/input_from_feature_columns/input_layer/Team_embedding/ExpandDimsWdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:         
о
]dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stackConst*
valueB"       *
dtype0*
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
░
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
°
Wdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_sliceStridedSliceOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Where]dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_1_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:         *
Index0*
T0	
░
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stackConst*
valueB"        *
dtype0*
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
▓
adnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Д
Ydnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1StridedSliceOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Where_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stackadnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_1adnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1/stack_2*
Index0*
T0	*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*'
_output_shapes
:         
х
Qdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/unstackUnpackNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast*
T0	*	
num*

axis *
_output_shapes
: : 
ц
Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/stackPackSdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/unstack:1*
T0	*

axis *
N*
_output_shapes
:
▓
Mdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/MulMulYdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_slice_1Odnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/stack*
T0	*'
_output_shapes
:         
й
_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╧
Mdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/SumSumMdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Mul_dnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Sum/reduction_indices*
T0	*#
_output_shapes
:         *

Tidx0*
	keep_dims( 
к
Mdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/AddAddWdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/strided_sliceMdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Sum*
T0	*#
_output_shapes
:         
╪
Pdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/GatherGatherQdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/ReshapeMdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Add*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:         
Д
Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_tableHashTableV2*
value_dtype0	*
_output_shapes
: *9
shared_name*(hash_table_vocab_list/teams.txt_32_-2_-1*
use_node_name_sharing( *
	key_dtype0*
	container 
б
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/ConstConst*
valueB	 R
         *
dtype0	*
_output_shapes
: 
┐
jdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init/asset_filepathConst*%
valueB Bvocab_list/teams.txt*
dtype0*
_output_shapes
: 
Ф
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_initInitializeTableFromTextFileV2Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_tablejdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init/asset_filepath*
	key_index■        *
value_index         *

vocab_size *
	delimiter	
Ш
Kdnn/input_from_feature_columns/input_layer/Team_embedding/hash_table_LookupLookupTableFindV2Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_tablePdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/GatherVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:         *	
Tin0
н
udnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
valueB"        *
dtype0*
_output_shapes
:
а
tdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
valueB
 *    *
dtype0*
_output_shapes
: 
в
vdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
valueB
 *є5>*
dtype0*
_output_shapes
: 
┐
dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaludnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
seed2 *
dtype0*
_output_shapes

:  *

seed 
Г
sdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMuldnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalvdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
_output_shapes

:  
ё
odnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normalAddsdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/multdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
_output_shapes

:  
н
Rdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0
VariableV2*
shape
:  *
dtype0*
_output_shapes

:  *
shared_name *e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
	container 
с
Ydnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/AssignAssignRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0odnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:  *
use_locking(
╟
Wdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/readIdentityRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
_output_shapes

:  
ж
\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
е
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Ь
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SliceSliceNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/begin[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice/size*
T0	*
Index0*
_output_shapes
:
а
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
╦
Udnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ProdProdVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SliceVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Const*

Tidx0*
	keep_dims( *
T0	*
_output_shapes
: 
б
_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GatherGatherNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather/indices*
Tparams0	*
validate_indices(*
_output_shapes
: *
Tindices0
╔
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast/xPackUdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ProdWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather*
T0	*

axis *
N*
_output_shapes
:
Ш
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshapeSparseReshapeOdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/WhereNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/CastWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast/x*-
_output_shapes
:         :
ю
gdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape/IdentityIdentityKdnn/input_from_feature_columns/input_layer/Team_embedding/hash_table_Lookup*
T0	*#
_output_shapes
:         
б
_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
х
]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqualGreaterEqualgdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape/Identity_dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:         
ч
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/WhereWhere]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/GreaterEqual*'
_output_shapes
:         
▒
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
╫
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ReshapeReshapeVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Where^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:         
¤
Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_1Gather^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshapeXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape*
Tindices0	*
Tparams0	*
validate_indices(*'
_output_shapes
:         
В
Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_2Gathergdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape/IdentityXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape*
Tindices0	*
Tparams0	*
validate_indices(*#
_output_shapes
:         
ь
Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/IdentityIdentity`dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
м
jdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
ы
xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsYdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_1Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Gather_2Ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Identityjdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:         :         :         :         
═
|dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
╧
~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
╧
~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
Э
vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicexdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows|dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1~dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*#
_output_shapes
:         
к
mdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/CastCastvdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:         *

DstT0
┴
odnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/UniqueUniquezdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*
out_idx0*2
_output_shapes 
:         :         
Ф
ydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherWdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/readodnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/Unique*
Tparams0*
validate_indices(*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*'
_output_shapes
:          *
Tindices0	
Ш
hdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparseSparseSegmentMeanydnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/embedding_lookupqdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/Unique:1mdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse/Cast*

Tidx0*
T0*'
_output_shapes
:          
▒
`dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
Г
Zdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1Reshapezdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2`dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
■
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/ShapeShapehdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse*
T0*
out_type0*
_output_shapes
:
о
ddnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
░
fdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
░
fdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
О
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_sliceStridedSliceVdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Shapeddnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stackfdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_1fdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Ъ
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
╥
Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stackPackXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stack/0^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/strided_slice*
T0*

axis *
N*
_output_shapes
:
▐
Udnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/TileTileZdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_1Vdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/stack*
T0
*0
_output_shapes
:                  *

Tmultiples0
Д
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/zeros_like	ZerosLikehdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:          
к
Pdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weightsSelectUdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Tile[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/zeros_likehdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:          
у
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast_1CastNdnn/input_from_feature_columns/input_layer/Team_embedding/to_sparse_input/Cast*

SrcT0	*
_output_shapes
:*

DstT0
и
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
з
]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
л
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1SliceWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Cast_1^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/begin]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1/size*
T0*
Index0*
_output_shapes
:
ш
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Shape_1ShapePdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights*
T0*
out_type0*
_output_shapes
:
и
^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
░
]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/sizeConst*
valueB:
         *
dtype0*
_output_shapes
:
м
Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2SliceXdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Shape_1^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/begin]dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2/size*
T0*
Index0*
_output_shapes
:
Ю
\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
п
Wdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concatConcatV2Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_1Xdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Slice_2\dnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
╨
Zdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_2ReshapePdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weightsWdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/concat*
T0*
Tshape0*'
_output_shapes
:          
┘
?dnn/input_from_feature_columns/input_layer/Team_embedding/ShapeShapeZdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_2*
T0*
out_type0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ы
Gdnn/input_from_feature_columns/input_layer/Team_embedding/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/Team_embedding/ShapeMdnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stackOdnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/Team_embedding/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Л
Idnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shape/1Const*
value	B : *
dtype0*
_output_shapes
: 
Э
Gdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/Team_embedding/strided_sliceIdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
▒
Adnn/input_from_feature_columns/input_layer/Team_embedding/ReshapeReshapeZdnn/input_from_feature_columns/input_layer/Team_embedding/Team_embedding_weights/Reshape_2Gdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:          
П
Ddnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
ц
@dnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims
ExpandDimsIteratorGetNext:13Ddnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╚
=dnn/input_from_feature_columns/input_layer/Touchdowns/ToFloatCast@dnn/input_from_feature_columns/input_layer/Touchdowns/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
╕
;dnn/input_from_feature_columns/input_layer/Touchdowns/ShapeShape=dnn/input_from_feature_columns/input_layer/Touchdowns/ToFloat*
T0*
out_type0*
_output_shapes
:
У
Idnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Х
Kdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
Cdnn/input_from_feature_columns/input_layer/Touchdowns/strided_sliceStridedSlice;dnn/input_from_feature_columns/input_layer/Touchdowns/ShapeIdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stackKdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_1Kdnn/input_from_feature_columns/input_layer/Touchdowns/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
З
Ednn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
С
Cdnn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shapePackCdnn/input_from_feature_columns/input_layer/Touchdowns/strided_sliceEdnn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
М
=dnn/input_from_feature_columns/input_layer/Touchdowns/ReshapeReshape=dnn/input_from_feature_columns/input_layer/Touchdowns/ToFloatCdnn/input_from_feature_columns/input_layer/Touchdowns/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
К
?dnn/input_from_feature_columns/input_layer/Yards/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
▄
;dnn/input_from_feature_columns/input_layer/Yards/ExpandDims
ExpandDimsIteratorGetNext:14?dnn/input_from_feature_columns/input_layer/Yards/ExpandDims/dim*
T0*'
_output_shapes
:         *

Tdim0
╛
8dnn/input_from_feature_columns/input_layer/Yards/ToFloatCast;dnn/input_from_feature_columns/input_layer/Yards/ExpandDims*

SrcT0*'
_output_shapes
:         *

DstT0
о
6dnn/input_from_feature_columns/input_layer/Yards/ShapeShape8dnn/input_from_feature_columns/input_layer/Yards/ToFloat*
T0*
out_type0*
_output_shapes
:
О
Ddnn/input_from_feature_columns/input_layer/Yards/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ю
>dnn/input_from_feature_columns/input_layer/Yards/strided_sliceStridedSlice6dnn/input_from_feature_columns/input_layer/Yards/ShapeDdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stackFdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_1Fdnn/input_from_feature_columns/input_layer/Yards/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
В
@dnn/input_from_feature_columns/input_layer/Yards/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
В
>dnn/input_from_feature_columns/input_layer/Yards/Reshape/shapePack>dnn/input_from_feature_columns/input_layer/Yards/strided_slice@dnn/input_from_feature_columns/input_layer/Yards/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
¤
8dnn/input_from_feature_columns/input_layer/Yards/ReshapeReshape8dnn/input_from_feature_columns/input_layer/Yards/ToFloat>dnn/input_from_feature_columns/input_layer/Yards/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
я
1dnn/input_from_feature_columns/input_layer/concatConcatV2;dnn/input_from_feature_columns/input_layer/Attempts/Reshape>dnn/input_from_feature_columns/input_layer/Completions/ReshapeGdnn/input_from_feature_columns/input_layer/Conference_embedding/Reshape<dnn/input_from_feature_columns/input_layer/DraftYear/Reshape>dnn/input_from_feature_columns/input_layer/GamesPlayed/Reshape:dnn/input_from_feature_columns/input_layer/Heisman/Reshape@dnn/input_from_feature_columns/input_layer/Interceptions/Reshape7dnn/input_from_feature_columns/input_layer/Pick/Reshape8dnn/input_from_feature_columns/input_layer/Round/Reshape?dnn/input_from_feature_columns/input_layer/RushAttempts/ReshapeAdnn/input_from_feature_columns/input_layer/RushTouchdowns/Reshape<dnn/input_from_feature_columns/input_layer/RushYards/ReshapeAdnn/input_from_feature_columns/input_layer/Team_embedding/Reshape=dnn/input_from_feature_columns/input_layer/Touchdowns/Reshape8dnn/input_from_feature_columns/input_layer/Yards/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:         9
┼
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB"9   2   *
dtype0*
_output_shapes
:
╖
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *■{r╛*
dtype0*
_output_shapes
: 
╖
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
valueB
 *■{r>*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:92*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
seed2 
Ъ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
м
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:92
Ю
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:92
╟
dnn/hiddenlayer_0/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:92*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
	container *
shape
:92
У
&dnn/hiddenlayer_0/kernel/part_0/AssignAssigndnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:92
о
$dnn/hiddenlayer_0/kernel/part_0/readIdentitydnn/hiddenlayer_0/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:92
о
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
valueB2*    *
dtype0*
_output_shapes
:2
╗
dnn/hiddenlayer_0/bias/part_0
VariableV2*
shape:2*
dtype0*
_output_shapes
:2*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
	container 
■
$dnn/hiddenlayer_0/bias/part_0/AssignAssigndnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:2*
use_locking(
д
"dnn/hiddenlayer_0/bias/part_0/readIdentitydnn/hiddenlayer_0/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:2
s
dnn/hiddenlayer_0/kernelIdentity$dnn/hiddenlayer_0/kernel/part_0/read*
T0*
_output_shapes

:92
╟
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
transpose_b( *
T0*'
_output_shapes
:         2*
transpose_a( 
k
dnn/hiddenlayer_0/biasIdentity"dnn/hiddenlayer_0/bias/part_0/read*
T0*
_output_shapes
:2
Я
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:         2
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:         2
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*'
_output_shapes
:         2
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*'
_output_shapes
:         2*

DstT0
h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Н
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
а
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
л
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
┼
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB"2   d   *
dtype0*
_output_shapes
:
╖
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *═╠L╛*
dtype0*
_output_shapes
: 
╖
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
seed2 *
dtype0*
_output_shapes

:2d*

seed 
Ъ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
м
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:2d
Ю
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:2d
╟
dnn/hiddenlayer_1/kernel/part_0
VariableV2*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
	container *
shape
:2d*
dtype0*
_output_shapes

:2d*
shared_name 
У
&dnn/hiddenlayer_1/kernel/part_0/AssignAssigndnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:2d*
use_locking(
о
$dnn/hiddenlayer_1/kernel/part_0/readIdentitydnn/hiddenlayer_1/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:2d
о
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
valueBd*    *
dtype0*
_output_shapes
:d
╗
dnn/hiddenlayer_1/bias/part_0
VariableV2*
dtype0*
_output_shapes
:d*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
	container *
shape:d
■
$dnn/hiddenlayer_1/bias/part_0/AssignAssigndnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:d
д
"dnn/hiddenlayer_1/bias/part_0/readIdentitydnn/hiddenlayer_1/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:d
s
dnn/hiddenlayer_1/kernelIdentity$dnn/hiddenlayer_1/kernel/part_0/read*
T0*
_output_shapes

:2d
м
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
transpose_b( *
T0*'
_output_shapes
:         d*
transpose_a( 
k
dnn/hiddenlayer_1/biasIdentity"dnn/hiddenlayer_1/bias/part_0/read*
T0*
_output_shapes
:d
Я
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*'
_output_shapes
:         d
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:         d
]
dnn/zero_fraction_1/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ж
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*
T0*'
_output_shapes
:         d
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*'
_output_shapes
:         d*

DstT0
j
dnn/zero_fraction_1/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
У
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
┼
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB"d   2   *
dtype0*
_output_shapes
:
╖
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB
 *═╠L╛*
dtype0*
_output_shapes
: 
╖
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
valueB
 *═╠L>*
dtype0*
_output_shapes
: 
Ю
Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:d2*

seed *
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
seed2 
Ъ
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
м
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:d2
Ю
:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:d2
╟
dnn/hiddenlayer_2/kernel/part_0
VariableV2*
dtype0*
_output_shapes

:d2*
shared_name *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
	container *
shape
:d2
У
&dnn/hiddenlayer_2/kernel/part_0/AssignAssigndnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(*
_output_shapes

:d2*
use_locking(
о
$dnn/hiddenlayer_2/kernel/part_0/readIdentitydnn/hiddenlayer_2/kernel/part_0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:d2
о
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
valueB2*    *
dtype0*
_output_shapes
:2
╗
dnn/hiddenlayer_2/bias/part_0
VariableV2*
shape:2*
dtype0*
_output_shapes
:2*
shared_name *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
	container 
■
$dnn/hiddenlayer_2/bias/part_0/AssignAssigndnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes
:2*
use_locking(
д
"dnn/hiddenlayer_2/bias/part_0/readIdentitydnn/hiddenlayer_2/bias/part_0*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:2
s
dnn/hiddenlayer_2/kernelIdentity$dnn/hiddenlayer_2/kernel/part_0/read*
T0*
_output_shapes

:d2
м
dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
transpose_b( *
T0*'
_output_shapes
:         2*
transpose_a( 
k
dnn/hiddenlayer_2/biasIdentity"dnn/hiddenlayer_2/bias/part_0/read*
T0*
_output_shapes
:2
Я
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
T0*
data_formatNHWC*'
_output_shapes
:         2
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:         2
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ж
dnn/zero_fraction_2/EqualEqualdnn/hiddenlayer_2/Reludnn/zero_fraction_2/zero*
T0*'
_output_shapes
:         2
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*'
_output_shapes
:         2*

DstT0
j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
У
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
а
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
н
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 
У
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
T0*
_output_shapes
: 
╖
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB"2      *
dtype0*
_output_shapes
:
й
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *SЭп╛*
dtype0*
_output_shapes
: 
й
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
valueB
 *SЭп>*
dtype0*
_output_shapes
: 
Й
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:2*

seed *
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
seed2 
■
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
Р
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:2
В
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:2
╣
dnn/logits/kernel/part_0
VariableV2*
shared_name *+
_class!
loc:@dnn/logits/kernel/part_0*
	container *
shape
:2*
dtype0*
_output_shapes

:2
ў
dnn/logits/kernel/part_0/AssignAssigndnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:2
Щ
dnn/logits/kernel/part_0/readIdentitydnn/logits/kernel/part_0*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:2
а
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
valueB*    *
dtype0*
_output_shapes
:
н
dnn/logits/bias/part_0
VariableV2*
shared_name *)
_class
loc:@dnn/logits/bias/part_0*
	container *
shape:*
dtype0*
_output_shapes
:
т
dnn/logits/bias/part_0/AssignAssigndnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:
П
dnn/logits/bias/part_0/readIdentitydnn/logits/bias/part_0*
T0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:
e
dnn/logits/kernelIdentitydnn/logits/kernel/part_0/read*
T0*
_output_shapes

:2
Ю
dnn/logits/MatMulMatMuldnn/hiddenlayer_2/Reludnn/logits/kernel*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
]
dnn/logits/biasIdentitydnn/logits/bias/part_0/read*
T0*
_output_shapes
:
К
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*'
_output_shapes
:         
]
dnn/zero_fraction_3/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
dnn/zero_fraction_3/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_3/zero*
T0*'
_output_shapes
:         
|
dnn/zero_fraction_3/CastCastdnn/zero_fraction_3/Equal*

SrcT0
*'
_output_shapes
:         *

DstT0
j
dnn/zero_fraction_3/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
У
dnn/zero_fraction_3/MeanMeandnn/zero_fraction_3/Castdnn/zero_fraction_3/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Т
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Я
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/Mean*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
Б
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
T0*
out_type0*
_output_shapes
:
n
,dnn/head/predictions/logits/assert_rank/rankConst*
value	B :*
dtype0*
_output_shapes
: 
^
Vdnn/head/predictions/logits/assert_rank/assert_type/statically_determined_correct_typeNoOp
O
Gdnn/head/predictions/logits/assert_rank/static_checks_determined_all_okNoOp
├
/dnn/head/predictions/logits/strided_slice/stackConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
┼
1dnn/head/predictions/logits/strided_slice/stack_1ConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
┼
1dnn/head/predictions/logits/strided_slice/stack_2ConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
Е
)dnn/head/predictions/logits/strided_sliceStridedSlice!dnn/head/predictions/logits/Shape/dnn/head/predictions/logits/strided_slice/stack1dnn/head/predictions/logits/strided_slice/stack_11dnn/head/predictions/logits/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
╢
*dnn/head/predictions/logits/assert_equal/xConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
п
.dnn/head/predictions/logits/assert_equal/EqualEqual*dnn/head/predictions/logits/assert_equal/x)dnn/head/predictions/logits/strided_slice*
T0*
_output_shapes
: 
╗
.dnn/head/predictions/logits/assert_equal/ConstConstH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok*
valueB *
dtype0*
_output_shapes
: 
╚
,dnn/head/predictions/logits/assert_equal/AllAll.dnn/head/predictions/logits/assert_equal/Equal.dnn/head/predictions/logits/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
о
6dnn/head/predictions/logits/assert_equal/Assert/AssertAssert,dnn/head/predictions/logits/assert_equal/All!dnn/head/predictions/logits/Shape*

T
2*
	summarize
Ё
dnn/head/predictions/logitsIdentitydnn/logits/BiasAddH^dnn/head/predictions/logits/assert_rank/static_checks_determined_all_ok7^dnn/head/predictions/logits/assert_equal/Assert/Assert*
T0*'
_output_shapes
:         
w
dnn/head/predictions/logisticSigmoiddnn/head/predictions/logits*
T0*'
_output_shapes
:         
{
dnn/head/predictions/zeros_like	ZerosLikednn/head/predictions/logits*
T0*'
_output_shapes
:         
l
*dnn/head/predictions/two_class_logits/axisConst*
value	B :*
dtype0*
_output_shapes
: 
т
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/head/predictions/logits*dnn/head/predictions/two_class_logits/axis*

Tidx0*
T0*
N*'
_output_shapes
:         
Ж
"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:         
g
%dnn/head/predictions/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
└
dnn/head/predictions/ArgMaxArgMax%dnn/head/predictions/two_class_logits%dnn/head/predictions/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
s
"dnn/head/predictions/classes/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
и
dnn/head/predictions/classesReshapednn/head/predictions/ArgMax"dnn/head/predictions/classes/shape*
T0	*
Tshape0*'
_output_shapes
:         
┌
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/classes*
width         *'
_output_shapes
:         *
	precision         *
shortest( *
T0	*

fill *

scientific( 
s
(dnn/head/maybe_expand_dim/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
о
$dnn/head/maybe_expand_dim/ExpandDims
ExpandDimsIteratorGetNext:15(dnn/head/maybe_expand_dim/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
y
dnn/head/labels/ShapeShape$dnn/head/maybe_expand_dim/ExpandDims*
T0*
out_type0*
_output_shapes
:
b
 dnn/head/labels/assert_rank/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Е
!dnn/head/labels/assert_rank/ShapeShape$dnn/head/maybe_expand_dim/ExpandDims*
T0*
out_type0*
_output_shapes
:
R
Jdnn/head/labels/assert_rank/assert_type/statically_determined_correct_typeNoOp
C
;dnn/head/labels/assert_rank/static_checks_determined_all_okNoOp
л
#dnn/head/labels/strided_slice/stackConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
н
%dnn/head/labels/strided_slice/stack_1Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
н
%dnn/head/labels/strided_slice/stack_2Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB:*
dtype0*
_output_shapes
:
╔
dnn/head/labels/strided_sliceStridedSlicednn/head/labels/Shape#dnn/head/labels/strided_slice/stack%dnn/head/labels/strided_slice/stack_1%dnn/head/labels/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ю
dnn/head/labels/assert_equal/xConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
value	B :*
dtype0*
_output_shapes
: 
Л
"dnn/head/labels/assert_equal/EqualEqualdnn/head/labels/assert_equal/xdnn/head/labels/strided_slice*
T0*
_output_shapes
: 
г
"dnn/head/labels/assert_equal/ConstConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*
valueB *
dtype0*
_output_shapes
: 
д
 dnn/head/labels/assert_equal/AllAll"dnn/head/labels/assert_equal/Equal"dnn/head/labels/assert_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
╠
)dnn/head/labels/assert_equal/Assert/ConstConst<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*5
value,B* B$labels shape must be [batch_size, 1]*
dtype0*
_output_shapes
: 
╒
+dnn/head/labels/assert_equal/Assert/Const_1Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
╤
+dnn/head/labels/assert_equal/Assert/Const_2Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*8
value/B- B'x (dnn/head/labels/assert_equal/x:0) = *
dtype0*
_output_shapes
: 
╨
+dnn/head/labels/assert_equal/Assert/Const_3Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*7
value.B, B&y (dnn/head/labels/strided_slice:0) = *
dtype0*
_output_shapes
: 
╘
1dnn/head/labels/assert_equal/Assert/Assert/data_0Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*5
value,B* B$labels shape must be [batch_size, 1]*
dtype0*
_output_shapes
: 
█
1dnn/head/labels/assert_equal/Assert/Assert/data_1Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*<
value3B1 B+Condition x == y did not hold element-wise:*
dtype0*
_output_shapes
: 
╫
1dnn/head/labels/assert_equal/Assert/Assert/data_2Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*8
value/B- B'x (dnn/head/labels/assert_equal/x:0) = *
dtype0*
_output_shapes
: 
╓
1dnn/head/labels/assert_equal/Assert/Assert/data_4Const<^dnn/head/labels/assert_rank/static_checks_determined_all_ok*7
value.B, B&y (dnn/head/labels/strided_slice:0) = *
dtype0*
_output_shapes
: 
Г
*dnn/head/labels/assert_equal/Assert/AssertAssert dnn/head/labels/assert_equal/All1dnn/head/labels/assert_equal/Assert/Assert/data_01dnn/head/labels/assert_equal/Assert/Assert/data_11dnn/head/labels/assert_equal/Assert/Assert/data_2dnn/head/labels/assert_equal/x1dnn/head/labels/assert_equal/Assert/Assert/data_4dnn/head/labels/strided_slice*
T

2*
	summarize
▐
dnn/head/labelsIdentity$dnn/head/maybe_expand_dim/ExpandDims<^dnn/head/labels/assert_rank/static_checks_determined_all_ok+^dnn/head/labels/assert_equal/Assert/Assert*
T0*'
_output_shapes
:         
j
dnn/head/ToFloatCastdnn/head/labels*

SrcT0*'
_output_shapes
:         *

DstT0
`
dnn/head/assert_range/ConstConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
П
&dnn/head/assert_range/assert_less/LessLessdnn/head/ToFloatdnn/head/assert_range/Const*
T0*'
_output_shapes
:         
x
'dnn/head/assert_range/assert_less/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
▓
%dnn/head/assert_range/assert_less/AllAll&dnn/head/assert_range/assert_less/Less'dnn/head/assert_range/assert_less/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
Й
.dnn/head/assert_range/assert_less/Assert/ConstConst*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
┤
0dnn/head/assert_range/assert_less/Assert/Const_1Const*T
valueKBI BCCondition x < y did not hold element-wise:x (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
Х
0dnn/head/assert_range/assert_less/Assert/Const_2Const*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
╢
;dnn/head/assert_range/assert_less/Assert/AssertGuard/SwitchSwitch%dnn/head/assert_range/assert_less/All%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: : 
й
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_tIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
з
=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_fIdentity;dnn/head/assert_range/assert_less/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
Р
<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_idIdentity%dnn/head/assert_range/assert_less/All*
T0
*
_output_shapes
: 
Б
9dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOpNoOp>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t
┴
Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependencyIdentity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:^dnn/head/assert_range/assert_less/Assert/AssertGuard/NoOp*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t*
_output_shapes
: 
▌
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*+
value"B  BLabel IDs must < n_classes*
dtype0*
_output_shapes
: 
Ж
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*T
valueKBI BCCondition x < y did not hold element-wise:x (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
ч
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_3Const>^dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*5
value,B* B$y (dnn/head/assert_range/Const:0) = *
dtype0*
_output_shapes
: 
О
Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchSwitch%dnn/head/assert_range/assert_less/All<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0
*8
_class.
,*loc:@dnn/head/assert_range/assert_less/All*
_output_shapes
: : 
И
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloat<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0*#
_class
loc:@dnn/head/ToFloat*:
_output_shapes(
&:         :         
№
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/assert_range/Const<dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id*
T0*.
_class$
" loc:@dnn/head/assert_range/Const*
_output_shapes
: : 
В
;dnn/head/assert_range/assert_less/Assert/AssertGuard/AssertAssertBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/SwitchBdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1Bdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_3Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
┼
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Identity=dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f<^dnn/head/assert_range/assert_less/Assert/AssertGuard/Assert*
T0
*P
_classF
DBloc:@dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f*
_output_shapes
: 
Г
:dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeMergeIdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1Gdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
t
/dnn/head/assert_range/assert_non_negative/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╟
Ednn/head/assert_range/assert_non_negative/assert_less_equal/LessEqual	LessEqual/dnn/head/assert_range/assert_non_negative/Constdnn/head/ToFloat*
T0*'
_output_shapes
:         
Т
Adnn/head/assert_range/assert_non_negative/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllAllEdnn/head/assert_range/assert_non_negative/assert_less_equal/LessEqualAdnn/head/assert_range/assert_non_negative/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
Ь
Hdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/ConstConst*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
╢
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_1Const*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
д
Jdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/Const_2Const**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
Д
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/All?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: : 
▌
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
█
Wdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fIdentityUdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
─
Vdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_idIdentity?dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
T0
*
_output_shapes
: 
╡
Sdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t
й
adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_tT^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
К
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*$
valueB BLabel IDs must >= 0*
dtype0*
_output_shapes
: 
в
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*<
value3B1 B+Condition x >= 0 did not hold element-wise:*
dtype0*
_output_shapes
: 
Р
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2ConstX^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f**
value!B Bx (dnn/head/ToFloat:0) = *
dtype0*
_output_shapes
: 
Ў
\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch?dnn/head/assert_range/assert_non_negative/assert_less_equal/AllVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*R
_classH
FDloc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/All*
_output_shapes
: : 
╝
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/ToFloatVdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id*
T0*#
_class
loc:@dnn/head/ToFloat*:
_output_shapes(
&:         :         
╫
Udnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1\dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1*
T
2*
	summarize
н
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityWdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_fV^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*j
_class`
^\loc:@dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
╤
Tdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/MergeMergecdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1adnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
В
dnn/head/assert_range/IdentityIdentitydnn/head/ToFloat;^dnn/head/assert_range/assert_less/Assert/AssertGuard/MergeU^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Merge*
T0*'
_output_shapes
:         
}
!dnn/head/logistic_loss/zeros_like	ZerosLikednn/head/predictions/logits*
T0*'
_output_shapes
:         
е
#dnn/head/logistic_loss/GreaterEqualGreaterEqualdnn/head/predictions/logits!dnn/head/logistic_loss/zeros_like*
T0*'
_output_shapes
:         
╛
dnn/head/logistic_loss/SelectSelect#dnn/head/logistic_loss/GreaterEqualdnn/head/predictions/logits!dnn/head/logistic_loss/zeros_like*
T0*'
_output_shapes
:         
p
dnn/head/logistic_loss/NegNegdnn/head/predictions/logits*
T0*'
_output_shapes
:         
╣
dnn/head/logistic_loss/Select_1Select#dnn/head/logistic_loss/GreaterEqualdnn/head/logistic_loss/Negdnn/head/predictions/logits*
T0*'
_output_shapes
:         
Р
dnn/head/logistic_loss/mulMuldnn/head/predictions/logitsdnn/head/assert_range/Identity*
T0*'
_output_shapes
:         
О
dnn/head/logistic_loss/subSubdnn/head/logistic_loss/Selectdnn/head/logistic_loss/mul*
T0*'
_output_shapes
:         
t
dnn/head/logistic_loss/ExpExpdnn/head/logistic_loss/Select_1*
T0*'
_output_shapes
:         
s
dnn/head/logistic_loss/Log1pLog1pdnn/head/logistic_loss/Exp*
T0*'
_output_shapes
:         
Й
dnn/head/logistic_lossAdddnn/head/logistic_loss/subdnn/head/logistic_loss/Log1p*
T0*'
_output_shapes
:         
x
3dnn/head/weighted_loss/assert_broadcastable/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
|
9dnn/head/weighted_loss/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
z
8dnn/head/weighted_loss/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
О
8dnn/head/weighted_loss/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
T0*
out_type0*
_output_shapes
:
y
7dnn/head/weighted_loss/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
O
Gdnn/head/weighted_loss/assert_broadcastable/static_scalar_check_successNoOp
▒
"dnn/head/weighted_loss/ToFloat_1/xConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
П
dnn/head/weighted_loss/MulMuldnn/head/logistic_loss"dnn/head/weighted_loss/ToFloat_1/x*
T0*'
_output_shapes
:         
╖
dnn/head/weighted_loss/ConstConstH^dnn/head/weighted_loss/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
Щ
dnn/head/weighted_loss/SumSumdnn/head/weighted_loss/Muldnn/head/weighted_loss/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
z
5dnn/head/metrics/label/mean/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
Pdnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
С
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
н
Odnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
Р
Ndnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
f
^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
№
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ShapeShapednn/head/assert_range/Identity_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
у
=dnn/head/metrics/label/mean/broadcast_weights/ones_like/ConstConst_^dnn/head/metrics/label/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
я
7dnn/head/metrics/label/mean/broadcast_weights/ones_likeFill=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Shape=dnn/head/metrics/label/mean/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
╓
-dnn/head/metrics/label/mean/broadcast_weightsMul5dnn/head/metrics/label/mean/broadcast_weights/weights7dnn/head/metrics/label/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
о
3dnn/head/metrics/label/mean/total/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
!dnn/head/metrics/label/mean/total
VariableV2*
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
	container *
shape: *
dtype0*
_output_shapes
: 
К
(dnn/head/metrics/label/mean/total/AssignAssign!dnn/head/metrics/label/mean/total3dnn/head/metrics/label/mean/total/Initializer/zeros*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
validate_shape(*
_output_shapes
: *
use_locking(
м
&dnn/head/metrics/label/mean/total/readIdentity!dnn/head/metrics/label/mean/total*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: 
о
3dnn/head/metrics/label/mean/count/Initializer/zerosConst*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
!dnn/head/metrics/label/mean/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
	container *
shape: 
К
(dnn/head/metrics/label/mean/count/AssignAssign!dnn/head/metrics/label/mean/count3dnn/head/metrics/label/mean/count/Initializer/zeros*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
validate_shape(*
_output_shapes
: *
use_locking(
м
&dnn/head/metrics/label/mean/count/readIdentity!dnn/head/metrics/label/mean/count*
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: 
┐
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape-dnn/head/metrics/label/mean/broadcast_weights*
T0*
out_type0*
_output_shapes
:
У
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
п
Qdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/assert_range/Identity*
T0*
out_type0*
_output_shapes
:
Т
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Т
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
Э
Ndnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
з
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
ч
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentity\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
х
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityZdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
╪
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Щ
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
├
Бdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchPdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*c
_classY
WUloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
╟
Гdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
о
zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualБdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchГdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Щ
tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Ы
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityvdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Щ
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitytdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
Ю
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
╥
Нdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
▄
Рdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
╕
Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchРdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Ї
Йdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsФdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Нdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
┘
Оdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╩
Оdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
▄
Иdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillОdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeОdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
╞
Кdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ї
Еdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Йdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsИdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeКdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*
_output_shapes

:
╘
Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
р
Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╜
Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchТdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
·
Лdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЦdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:*

Tdim0
┴
Чdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationЛdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Еdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
set_operationa-b*
T0*
validate_indices(*<
_output_shapes*
(:         :         :
х
Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeЩdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
╝
Аdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstw^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
╜
~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualАdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xПdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
и
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchzdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankudnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*П
_classД
Бloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
а
sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergevdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1~dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
у
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergesdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
▓
Jdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ы
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╝
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
Ъ
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
н
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
Ч
Ldnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
║
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
с
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
▀
Ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityWdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
р
Xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
╣
Udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
▒
cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tV^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
в
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Й
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
к
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*@
value7B5 B/dnn/head/metrics/label/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
И
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
Ы
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*1
value(B& B dnn/head/assert_range/Identity:0*
dtype0*
_output_shapes
: 
Е
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstZ^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
о
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
к
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchRdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*e
_class[
YWloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
и
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchQdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shapeXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*d
_classZ
XVloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Ъ
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchNdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*a
_classW
USloc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
╠
Wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssert^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
╡
ednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityYdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fX^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*l
_classb
`^loc:@dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
╫
Vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1cdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Ў
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/assert_range/IdentityW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
▌
?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/ConstConstW^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ї
9dnn/head/metrics/label/mean/broadcast_weights_1/ones_likeFill?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Shape?dnn/head/metrics/label/mean/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
╥
/dnn/head/metrics/label/mean/broadcast_weights_1Mul-dnn/head/metrics/label/mean/broadcast_weights9dnn/head/metrics/label/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
й
dnn/head/metrics/label/mean/MulMuldnn/head/assert_range/Identity/dnn/head/metrics/label/mean/broadcast_weights_1*
T0*'
_output_shapes
:         
r
!dnn/head/metrics/label/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╕
dnn/head/metrics/label/mean/SumSum/dnn/head/metrics/label/mean/broadcast_weights_1!dnn/head/metrics/label/mean/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
t
#dnn/head/metrics/label/mean/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
м
!dnn/head/metrics/label/mean/Sum_1Sumdnn/head/metrics/label/mean/Mul#dnn/head/metrics/label/mean/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
т
%dnn/head/metrics/label/mean/AssignAdd	AssignAdd!dnn/head/metrics/label/mean/total!dnn/head/metrics/label/mean/Sum_1*
use_locking( *
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
_output_shapes
: 
Д
'dnn/head/metrics/label/mean/AssignAdd_1	AssignAdd!dnn/head/metrics/label/mean/countdnn/head/metrics/label/mean/Sum ^dnn/head/metrics/label/mean/Mul*
use_locking( *
T0*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
_output_shapes
: 
j
%dnn/head/metrics/label/mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ю
#dnn/head/metrics/label/mean/GreaterGreater&dnn/head/metrics/label/mean/count/read%dnn/head/metrics/label/mean/Greater/y*
T0*
_output_shapes
: 
Я
#dnn/head/metrics/label/mean/truedivRealDiv&dnn/head/metrics/label/mean/total/read&dnn/head/metrics/label/mean/count/read*
T0*
_output_shapes
: 
h
#dnn/head/metrics/label/mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╗
!dnn/head/metrics/label/mean/valueSelect#dnn/head/metrics/label/mean/Greater#dnn/head/metrics/label/mean/truediv#dnn/head/metrics/label/mean/value/e*
T0*
_output_shapes
: 
l
'dnn/head/metrics/label/mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
г
%dnn/head/metrics/label/mean/Greater_1Greater'dnn/head/metrics/label/mean/AssignAdd_1'dnn/head/metrics/label/mean/Greater_1/y*
T0*
_output_shapes
: 
б
%dnn/head/metrics/label/mean/truediv_1RealDiv%dnn/head/metrics/label/mean/AssignAdd'dnn/head/metrics/label/mean/AssignAdd_1*
T0*
_output_shapes
: 
l
'dnn/head/metrics/label/mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╟
%dnn/head/metrics/label/mean/update_opSelect%dnn/head/metrics/label/mean/Greater_1%dnn/head/metrics/label/mean/truediv_1'dnn/head/metrics/label/mean/update_op/e*
T0*
_output_shapes
: 
▓
5dnn/head/metrics/average_loss/total/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
#dnn/head/metrics/average_loss/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
	container 
Т
*dnn/head/metrics/average_loss/total/AssignAssign#dnn/head/metrics/average_loss/total5dnn/head/metrics/average_loss/total/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
validate_shape(*
_output_shapes
: 
▓
(dnn/head/metrics/average_loss/total/readIdentity#dnn/head/metrics/average_loss/total*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: 
▓
5dnn/head/metrics/average_loss/count/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
#dnn/head/metrics/average_loss/count
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
	container 
Т
*dnn/head/metrics/average_loss/count/AssignAssign#dnn/head/metrics/average_loss/count5dnn/head/metrics/average_loss/count/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
validate_shape(*
_output_shapes
: 
▓
(dnn/head/metrics/average_loss/count/readIdentity#dnn/head/metrics/average_loss/count*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: 
h
#dnn/head/metrics/average_loss/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Х
Rdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
У
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
з
Qdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/logistic_loss*
T0*
out_type0*
_output_shapes
:
Т
Pdnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
h
`dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
°
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ShapeShapednn/head/logistic_lossa^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ч
?dnn/head/metrics/average_loss/broadcast_weights/ones_like/ConstConsta^dnn/head/metrics/average_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ї
9dnn/head/metrics/average_loss/broadcast_weights/ones_likeFill?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Shape?dnn/head/metrics/average_loss/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
╚
/dnn/head/metrics/average_loss/broadcast_weightsMul#dnn/head/metrics/average_loss/Const9dnn/head/metrics/average_loss/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
г
!dnn/head/metrics/average_loss/MulMuldnn/head/logistic_loss/dnn/head/metrics/average_loss/broadcast_weights*
T0*'
_output_shapes
:         
v
%dnn/head/metrics/average_loss/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
╛
!dnn/head/metrics/average_loss/SumSum/dnn/head/metrics/average_loss/broadcast_weights%dnn/head/metrics/average_loss/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
v
%dnn/head/metrics/average_loss/Const_2Const*
valueB"       *
dtype0*
_output_shapes
:
▓
#dnn/head/metrics/average_loss/Sum_1Sum!dnn/head/metrics/average_loss/Mul%dnn/head/metrics/average_loss/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
ъ
'dnn/head/metrics/average_loss/AssignAdd	AssignAdd#dnn/head/metrics/average_loss/total#dnn/head/metrics/average_loss/Sum_1*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
_output_shapes
: *
use_locking( 
О
)dnn/head/metrics/average_loss/AssignAdd_1	AssignAdd#dnn/head/metrics/average_loss/count!dnn/head/metrics/average_loss/Sum"^dnn/head/metrics/average_loss/Mul*
T0*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
_output_shapes
: *
use_locking( 
l
'dnn/head/metrics/average_loss/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
д
%dnn/head/metrics/average_loss/GreaterGreater(dnn/head/metrics/average_loss/count/read'dnn/head/metrics/average_loss/Greater/y*
T0*
_output_shapes
: 
е
%dnn/head/metrics/average_loss/truedivRealDiv(dnn/head/metrics/average_loss/total/read(dnn/head/metrics/average_loss/count/read*
T0*
_output_shapes
: 
j
%dnn/head/metrics/average_loss/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
├
#dnn/head/metrics/average_loss/valueSelect%dnn/head/metrics/average_loss/Greater%dnn/head/metrics/average_loss/truediv%dnn/head/metrics/average_loss/value/e*
T0*
_output_shapes
: 
n
)dnn/head/metrics/average_loss/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
й
'dnn/head/metrics/average_loss/Greater_1Greater)dnn/head/metrics/average_loss/AssignAdd_1)dnn/head/metrics/average_loss/Greater_1/y*
T0*
_output_shapes
: 
з
'dnn/head/metrics/average_loss/truediv_1RealDiv'dnn/head/metrics/average_loss/AssignAdd)dnn/head/metrics/average_loss/AssignAdd_1*
T0*
_output_shapes
: 
n
)dnn/head/metrics/average_loss/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╧
'dnn/head/metrics/average_loss/update_opSelect'dnn/head/metrics/average_loss/Greater_1'dnn/head/metrics/average_loss/truediv_1)dnn/head/metrics/average_loss/update_op/e*
T0*
_output_shapes
: 
[
dnn/head/metrics/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
|
dnn/head/metrics/CastCastdnn/head/predictions/classes*

SrcT0	*'
_output_shapes
:         *

DstT0
И
dnn/head/metrics/EqualEqualdnn/head/metrics/Castdnn/head/assert_range/Identity*
T0*'
_output_shapes
:         
y
dnn/head/metrics/ToFloatCastdnn/head/metrics/Equal*

SrcT0
*'
_output_shapes
:         *

DstT0
к
1dnn/head/metrics/accuracy/total/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
valueB
 *    *
dtype0*
_output_shapes
: 
╖
dnn/head/metrics/accuracy/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
	container 
В
&dnn/head/metrics/accuracy/total/AssignAssigndnn/head/metrics/accuracy/total1dnn/head/metrics/accuracy/total/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
validate_shape(*
_output_shapes
: 
ж
$dnn/head/metrics/accuracy/total/readIdentitydnn/head/metrics/accuracy/total*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: 
к
1dnn/head/metrics/accuracy/count/Initializer/zerosConst*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
valueB
 *    *
dtype0*
_output_shapes
: 
╖
dnn/head/metrics/accuracy/count
VariableV2*
shared_name *2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
	container *
shape: *
dtype0*
_output_shapes
: 
В
&dnn/head/metrics/accuracy/count/AssignAssigndnn/head/metrics/accuracy/count1dnn/head/metrics/accuracy/count/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
validate_shape(*
_output_shapes
: 
ж
$dnn/head/metrics/accuracy/count/readIdentitydnn/head/metrics/accuracy/count*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
С
Ndnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
П
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
е
Mdnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/metrics/ToFloat*
T0*
out_type0*
_output_shapes
:
О
Ldnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
d
\dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Є
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ShapeShapednn/head/metrics/ToFloat]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
▀
;dnn/head/metrics/accuracy/broadcast_weights/ones_like/ConstConst]^dnn/head/metrics/accuracy/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
щ
5dnn/head/metrics/accuracy/broadcast_weights/ones_likeFill;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Shape;dnn/head/metrics/accuracy/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
│
+dnn/head/metrics/accuracy/broadcast_weightsMuldnn/head/metrics/Const5dnn/head/metrics/accuracy/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
Э
dnn/head/metrics/accuracy/MulMuldnn/head/metrics/ToFloat+dnn/head/metrics/accuracy/broadcast_weights*
T0*'
_output_shapes
:         
p
dnn/head/metrics/accuracy/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
░
dnn/head/metrics/accuracy/SumSum+dnn/head/metrics/accuracy/broadcast_weightsdnn/head/metrics/accuracy/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
r
!dnn/head/metrics/accuracy/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
ж
dnn/head/metrics/accuracy/Sum_1Sumdnn/head/metrics/accuracy/Mul!dnn/head/metrics/accuracy/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
┌
#dnn/head/metrics/accuracy/AssignAdd	AssignAdddnn/head/metrics/accuracy/totaldnn/head/metrics/accuracy/Sum_1*
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
_output_shapes
: *
use_locking( 
·
%dnn/head/metrics/accuracy/AssignAdd_1	AssignAdddnn/head/metrics/accuracy/countdnn/head/metrics/accuracy/Sum^dnn/head/metrics/accuracy/Mul*
use_locking( *
T0*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
_output_shapes
: 
h
#dnn/head/metrics/accuracy/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
!dnn/head/metrics/accuracy/GreaterGreater$dnn/head/metrics/accuracy/count/read#dnn/head/metrics/accuracy/Greater/y*
T0*
_output_shapes
: 
Щ
!dnn/head/metrics/accuracy/truedivRealDiv$dnn/head/metrics/accuracy/total/read$dnn/head/metrics/accuracy/count/read*
T0*
_output_shapes
: 
f
!dnn/head/metrics/accuracy/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
│
dnn/head/metrics/accuracy/valueSelect!dnn/head/metrics/accuracy/Greater!dnn/head/metrics/accuracy/truediv!dnn/head/metrics/accuracy/value/e*
T0*
_output_shapes
: 
j
%dnn/head/metrics/accuracy/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Э
#dnn/head/metrics/accuracy/Greater_1Greater%dnn/head/metrics/accuracy/AssignAdd_1%dnn/head/metrics/accuracy/Greater_1/y*
T0*
_output_shapes
: 
Ы
#dnn/head/metrics/accuracy/truediv_1RealDiv#dnn/head/metrics/accuracy/AssignAdd%dnn/head/metrics/accuracy/AssignAdd_1*
T0*
_output_shapes
: 
j
%dnn/head/metrics/accuracy/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
┐
#dnn/head/metrics/accuracy/update_opSelect#dnn/head/metrics/accuracy/Greater_1#dnn/head/metrics/accuracy/truediv_1%dnn/head/metrics/accuracy/update_op/e*
T0*
_output_shapes
: 

:dnn/head/metrics/prediction/mean/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ш
Udnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ц
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
▒
Tdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Х
Sdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
k
cdnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
Е
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
э
Bdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ConstConstd^dnn/head/metrics/prediction/mean/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
■
<dnn/head/metrics/prediction/mean/broadcast_weights/ones_likeFillBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/ShapeBdnn/head/metrics/prediction/mean/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
х
2dnn/head/metrics/prediction/mean/broadcast_weightsMul:dnn/head/metrics/prediction/mean/broadcast_weights/weights<dnn/head/metrics/prediction/mean/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
╕
8dnn/head/metrics/prediction/mean/total/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
┼
&dnn/head/metrics/prediction/mean/total
VariableV2*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
Ю
-dnn/head/metrics/prediction/mean/total/AssignAssign&dnn/head/metrics/prediction/mean/total8dnn/head/metrics/prediction/mean/total/Initializer/zeros*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
validate_shape(*
_output_shapes
: *
use_locking(
╗
+dnn/head/metrics/prediction/mean/total/readIdentity&dnn/head/metrics/prediction/mean/total*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
_output_shapes
: 
╕
8dnn/head/metrics/prediction/mean/count/Initializer/zerosConst*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
┼
&dnn/head/metrics/prediction/mean/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
	container *
shape: 
Ю
-dnn/head/metrics/prediction/mean/count/AssignAssign&dnn/head/metrics/prediction/mean/count8dnn/head/metrics/prediction/mean/count/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
validate_shape(*
_output_shapes
: 
╗
+dnn/head/metrics/prediction/mean/count/readIdentity&dnn/head/metrics/prediction/mean/count*
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
_output_shapes
: 
╔
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shapeShape2dnn/head/metrics/prediction/mean/broadcast_weights*
T0*
out_type0*
_output_shapes
:
Ш
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
│
Vdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Ч
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Ч
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
м
Sdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarEqualUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar/xVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
╢
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalarSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
ё
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityadnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
я
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentity_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
т
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentitySdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
н
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
╫
Жdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchUdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*h
_class^
\Zloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
█
Иdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
╜
dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualЖdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchИdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
и
ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
е
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentity{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
г
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
и
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
▄
Тdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Ё
Хdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
╠
Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchХdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Г
Оdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЩdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Тdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
у
Уdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╘
Уdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ы
Нdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillУdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeУdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
╨
Пdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
И
Кdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Оdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsНdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeПdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
▐
Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Ї
Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╤
Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЧdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
Й
Рdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsЫdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:*

Tdim0
╨
Ьdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationРdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Кdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
я
Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeЮdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
╞
Еdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst|^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
═
Гdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualЕdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xФdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
╜
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankzdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Х
_classК
ЗДloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
░
xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMerge{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Гdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Є
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergexdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergecdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
╖
Odnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
а
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╞
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_2Const*E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
Я
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
▒
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
Ь
Qdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╔
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
ы
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
щ
^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentity\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
ъ
]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
├
Zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOp_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
┼
hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t[^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
м
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
У
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╣
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*E
value<B: B4dnn/head/metrics/prediction/mean/broadcast_weights:0*
dtype0*
_output_shapes
: 
Т
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
д
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
П
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Const_^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
┬
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitch^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
╛
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchWdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*j
_class`
^\loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╝
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchVdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*i
_class_
][loc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
о
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchSdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar]dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*f
_class\
ZXloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
Г	
\dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchcdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
╔
jdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identity^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f]^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*q
_classg
ecloc:@dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
ц
[dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergejdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1hdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
 
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistic\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
ч
Ddnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ConstConst\^dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Д
>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_likeFillDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/ShapeDdnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
с
4dnn/head/metrics/prediction/mean/broadcast_weights_1Mul2dnn/head/metrics/prediction/mean/broadcast_weights>dnn/head/metrics/prediction/mean/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
▓
$dnn/head/metrics/prediction/mean/MulMuldnn/head/predictions/logistic4dnn/head/metrics/prediction/mean/broadcast_weights_1*
T0*'
_output_shapes
:         
w
&dnn/head/metrics/prediction/mean/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╟
$dnn/head/metrics/prediction/mean/SumSum4dnn/head/metrics/prediction/mean/broadcast_weights_1&dnn/head/metrics/prediction/mean/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
y
(dnn/head/metrics/prediction/mean/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
╗
&dnn/head/metrics/prediction/mean/Sum_1Sum$dnn/head/metrics/prediction/mean/Mul(dnn/head/metrics/prediction/mean/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Ў
*dnn/head/metrics/prediction/mean/AssignAdd	AssignAdd&dnn/head/metrics/prediction/mean/total&dnn/head/metrics/prediction/mean/Sum_1*
use_locking( *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
_output_shapes
: 
Э
,dnn/head/metrics/prediction/mean/AssignAdd_1	AssignAdd&dnn/head/metrics/prediction/mean/count$dnn/head/metrics/prediction/mean/Sum%^dnn/head/metrics/prediction/mean/Mul*
use_locking( *
T0*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
_output_shapes
: 
o
*dnn/head/metrics/prediction/mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
н
(dnn/head/metrics/prediction/mean/GreaterGreater+dnn/head/metrics/prediction/mean/count/read*dnn/head/metrics/prediction/mean/Greater/y*
T0*
_output_shapes
: 
о
(dnn/head/metrics/prediction/mean/truedivRealDiv+dnn/head/metrics/prediction/mean/total/read+dnn/head/metrics/prediction/mean/count/read*
T0*
_output_shapes
: 
m
(dnn/head/metrics/prediction/mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╧
&dnn/head/metrics/prediction/mean/valueSelect(dnn/head/metrics/prediction/mean/Greater(dnn/head/metrics/prediction/mean/truediv(dnn/head/metrics/prediction/mean/value/e*
T0*
_output_shapes
: 
q
,dnn/head/metrics/prediction/mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
▓
*dnn/head/metrics/prediction/mean/Greater_1Greater,dnn/head/metrics/prediction/mean/AssignAdd_1,dnn/head/metrics/prediction/mean/Greater_1/y*
T0*
_output_shapes
: 
░
*dnn/head/metrics/prediction/mean/truediv_1RealDiv*dnn/head/metrics/prediction/mean/AssignAdd,dnn/head/metrics/prediction/mean/AssignAdd_1*
T0*
_output_shapes
: 
q
,dnn/head/metrics/prediction/mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
█
*dnn/head/metrics/prediction/mean/update_opSelect*dnn/head/metrics/prediction/mean/Greater_1*dnn/head/metrics/prediction/mean/truediv_1,dnn/head/metrics/prediction/mean/update_op/e*
T0*
_output_shapes
: 
m
(dnn/head/metrics/accuracy_baseline/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ы
&dnn/head/metrics/accuracy_baseline/subSub(dnn/head/metrics/accuracy_baseline/sub/x!dnn/head/metrics/label/mean/value*
T0*
_output_shapes
: 
Я
(dnn/head/metrics/accuracy_baseline/valueMaximum!dnn/head/metrics/label/mean/value&dnn/head/metrics/accuracy_baseline/sub*
T0*
_output_shapes
: 
o
*dnn/head/metrics/accuracy_baseline/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
г
(dnn/head/metrics/accuracy_baseline/sub_1Sub*dnn/head/metrics/accuracy_baseline/sub_1/x%dnn/head/metrics/label/mean/update_op*
T0*
_output_shapes
: 
й
,dnn/head/metrics/accuracy_baseline/update_opMaximum%dnn/head/metrics/label/mean/update_op(dnn/head/metrics/accuracy_baseline/sub_1*
T0*
_output_shapes
: 
В
dnn/head/metrics/auc/CastCastdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:         *

DstT0

s
.dnn/head/metrics/auc/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
М
Idnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
К
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
е
Hdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Й
Gdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
_
Wdnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
э
6dnn/head/metrics/auc/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logisticX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
╒
6dnn/head/metrics/auc/broadcast_weights/ones_like/ConstConstX^dnn/head/metrics/auc/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
┌
0dnn/head/metrics/auc/broadcast_weights/ones_likeFill6dnn/head/metrics/auc/broadcast_weights/ones_like/Shape6dnn/head/metrics/auc/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
┴
&dnn/head/metrics/auc/broadcast_weightsMul.dnn/head/metrics/auc/broadcast_weights/weights0dnn/head/metrics/auc/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
b
dnn/head/metrics/auc/Cast_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╢
6dnn/head/metrics/auc/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast_1/x*
T0*'
_output_shapes
:         
А
/dnn/head/metrics/auc/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╥
-dnn/head/metrics/auc/assert_greater_equal/AllAll6dnn/head/metrics/auc/assert_greater_equal/GreaterEqual/dnn/head/metrics/auc/assert_greater_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
Ф
6dnn/head/metrics/auc/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╩
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_1Const*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
Я
8dnn/head/metrics/auc/assert_greater_equal/Assert/Const_2Const*7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
╬
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/All-dnn/head/metrics/auc/assert_greater_equal/All*
T0
*
_output_shapes
: : 
╣
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╖
Ednn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fIdentityCdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
а
Ddnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_idIdentity-dnn/head/metrics/auc/assert_greater_equal/All*
T0
*
_output_shapes
: 
С
Adnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOpNoOpF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t
с
Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_tB^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
Ё
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
д
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
∙
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstF^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*7
value.B, B&y (dnn/head/metrics/auc/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
о
Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch-dnn/head/metrics/auc/assert_greater_equal/AllDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*@
_class6
42loc:@dnn/head/metrics/auc/assert_greater_equal/All*
_output_shapes
: : 
▓
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
Р
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast_1/xDdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/metrics/auc/Cast_1/x*
_output_shapes
: : 
║
Cdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/AssertAssertJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/SwitchJdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Jdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
х
Qdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityEdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_fD^dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*X
_classN
LJloc:@dnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
Ы
Bdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/MergeMergeQdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1Odnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
b
dnn/head/metrics/auc/Cast_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
н
0dnn/head/metrics/auc/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logisticdnn/head/metrics/auc/Cast_2/x*
T0*'
_output_shapes
:         
}
,dnn/head/metrics/auc/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
╞
*dnn/head/metrics/auc/assert_less_equal/AllAll0dnn/head/metrics/auc/assert_less_equal/LessEqual,dnn/head/metrics/auc/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
С
3dnn/head/metrics/auc/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╟
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_1Const*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
Ь
5dnn/head/metrics/auc/assert_less_equal/Assert/Const_2Const*7
value.B, B&y (dnn/head/metrics/auc/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
┼
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/All*dnn/head/metrics/auc/assert_less_equal/All*
T0
*
_output_shapes
: : 
│
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_tIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
▒
Bdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fIdentity@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ъ
Adnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_idIdentity*dnn/head/metrics/auc/assert_less_equal/All*
T0
*
_output_shapes
: 
Л
>dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOpNoOpC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t
╒
Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependencyIdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t?^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
ъ
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
Ю
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
є
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstC^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*7
value.B, B&y (dnn/head/metrics/auc/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
в
Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch*dnn/head/metrics/auc/assert_less_equal/AllAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*=
_class3
1/loc:@dnn/head/metrics/auc/assert_less_equal/All*
_output_shapes
: : 
м
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
К
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switchdnn/head/metrics/auc/Cast_2/xAdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/metrics/auc/Cast_2/x*
_output_shapes
: : 
е
@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/AssertAssertGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/SwitchGdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Gdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
┘
Ndnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentityBdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_fA^dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert*
T0
*U
_classK
IGloc:@dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
Т
?dnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/MergeMergeNdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1Ldnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
s
"dnn/head/metrics/auc/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
к
dnn/head/metrics/auc/ReshapeReshapednn/head/predictions/logistic"dnn/head/metrics/auc/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
u
$dnn/head/metrics/auc/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
к
dnn/head/metrics/auc/Reshape_1Reshapednn/head/metrics/auc/Cast$dnn/head/metrics/auc/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
v
dnn/head/metrics/auc/ShapeShapednn/head/metrics/auc/Reshape*
T0*
out_type0*
_output_shapes
:
r
(dnn/head/metrics/auc/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
t
*dnn/head/metrics/auc/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
т
"dnn/head/metrics/auc/strided_sliceStridedSlicednn/head/metrics/auc/Shape(dnn/head/metrics/auc/strided_slice/stack*dnn/head/metrics/auc/strided_slice/stack_1*dnn/head/metrics/auc/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Й
dnn/head/metrics/auc/ConstConst*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
dtype0*
_output_shapes	
:╚
m
#dnn/head/metrics/auc/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
д
dnn/head/metrics/auc/ExpandDims
ExpandDimsdnn/head/metrics/auc/Const#dnn/head/metrics/auc/ExpandDims/dim*
T0*
_output_shapes
:	╚*

Tdim0
^
dnn/head/metrics/auc/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
Ю
dnn/head/metrics/auc/stackPackdnn/head/metrics/auc/stack/0"dnn/head/metrics/auc/strided_slice*
T0*

axis *
N*
_output_shapes
:
г
dnn/head/metrics/auc/TileTilednn/head/metrics/auc/ExpandDimsdnn/head/metrics/auc/stack*

Tmultiples0*
T0*(
_output_shapes
:╚         
j
#dnn/head/metrics/auc/transpose/RankRankdnn/head/metrics/auc/Reshape*
T0*
_output_shapes
: 
f
$dnn/head/metrics/auc/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Х
"dnn/head/metrics/auc/transpose/subSub#dnn/head/metrics/auc/transpose/Rank$dnn/head/metrics/auc/transpose/sub/y*
T0*
_output_shapes
: 
l
*dnn/head/metrics/auc/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
l
*dnn/head/metrics/auc/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
╥
$dnn/head/metrics/auc/transpose/RangeRange*dnn/head/metrics/auc/transpose/Range/start#dnn/head/metrics/auc/transpose/Rank*dnn/head/metrics/auc/transpose/Range/delta*

Tidx0*
_output_shapes
:
Ъ
$dnn/head/metrics/auc/transpose/sub_1Sub"dnn/head/metrics/auc/transpose/sub$dnn/head/metrics/auc/transpose/Range*
T0*
_output_shapes
:
о
dnn/head/metrics/auc/transpose	Transposednn/head/metrics/auc/Reshape$dnn/head/metrics/auc/transpose/sub_1*
Tperm0*
T0*'
_output_shapes
:         
v
%dnn/head/metrics/auc/Tile_1/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
п
dnn/head/metrics/auc/Tile_1Tilednn/head/metrics/auc/transpose%dnn/head/metrics/auc/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
Т
dnn/head/metrics/auc/GreaterGreaterdnn/head/metrics/auc/Tile_1dnn/head/metrics/auc/Tile*
T0*(
_output_shapes
:╚         
u
dnn/head/metrics/auc/LogicalNot
LogicalNotdnn/head/metrics/auc/Greater*(
_output_shapes
:╚         
v
%dnn/head/metrics/auc/Tile_2/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
п
dnn/head/metrics/auc/Tile_2Tilednn/head/metrics/auc/Reshape_1%dnn/head/metrics/auc/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
v
!dnn/head/metrics/auc/LogicalNot_1
LogicalNotdnn/head/metrics/auc/Tile_2*(
_output_shapes
:╚         
▒
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeShape&dnn/head/metrics/auc/broadcast_weights*
T0*
out_type0*
_output_shapes
:
М
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
з
Jdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Л
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Л
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
И
Gdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarEqualIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar/xJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
Т
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
┘
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityUdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
╫
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentitySdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
╩
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
¤
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
ж
zdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchIdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*\
_classR
PNloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
к
|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rankTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
Ч
sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualzdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Д
mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_ranksdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
Н
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
Л
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentitymdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
Р
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentitysdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
─
Жdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
└
Йdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Ь
Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЙdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
▀
Вdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsНdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Жdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*
_output_shapes

:
╦
Зdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
╝
Зdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╟
Бdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillЗdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeЗdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
╕
Гdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
╫
~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Вdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsБdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeГdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
╞
Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
─
Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeTdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
б
Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЛdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
х
Дdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsПdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

:
л
Рdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationДdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:         :         :*
set_operationa-b
╫
Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeТdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
н
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstp^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
з
wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xИdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
К
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switchsdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Ж
_class|
zxloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
Л
ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeodnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
╬
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMergeldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
л
Cdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ф
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
о
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_2Const*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
У
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
е
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
Р
Ednn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
е
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
╙
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╤
Rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityPdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
╥
Qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
л
Ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
Х
\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tO^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
Ф
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
√
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Х
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*9
value0B. B(dnn/head/metrics/auc/broadcast_weights:0*
dtype0*
_output_shapes
: 
·
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
М
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
ў
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7ConstS^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
Т
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
О
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchKdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*^
_classT
RPloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
М
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchJdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shapeQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*]
_classS
QOloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
■
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchGdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalarQdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*Z
_classP
NLloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
 
Pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAssertWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchWdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
Щ
^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1IdentityRdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fQ^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*e
_class[
YWloc:@dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
┬
Odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMerge^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1\dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
ч
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logisticP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
╧
8dnn/head/metrics/auc/broadcast_weights_1/ones_like/ConstConstP^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
р
2dnn/head/metrics/auc/broadcast_weights_1/ones_likeFill8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Shape8dnn/head/metrics/auc/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
╜
(dnn/head/metrics/auc/broadcast_weights_1Mul&dnn/head/metrics/auc/broadcast_weights2dnn/head/metrics/auc/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
u
$dnn/head/metrics/auc/Reshape_2/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
╣
dnn/head/metrics/auc/Reshape_2Reshape(dnn/head/metrics/auc/broadcast_weights_1$dnn/head/metrics/auc/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
v
%dnn/head/metrics/auc/Tile_3/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
п
dnn/head/metrics/auc/Tile_3Tilednn/head/metrics/auc/Reshape_2%dnn/head/metrics/auc/Tile_3/multiples*
T0*(
_output_shapes
:╚         *

Tmultiples0
╝
5dnn/head/metrics/auc/true_positives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╔
#dnn/head/metrics/auc/true_positives
VariableV2*
shared_name *6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚
Ч
*dnn/head/metrics/auc/true_positives/AssignAssign#dnn/head/metrics/auc/true_positives5dnn/head/metrics/auc/true_positives/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
validate_shape(*
_output_shapes	
:╚
╖
(dnn/head/metrics/auc/true_positives/readIdentity#dnn/head/metrics/auc/true_positives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
_output_shapes	
:╚
Т
dnn/head/metrics/auc/LogicalAnd
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/Greater*(
_output_shapes
:╚         
Й
dnn/head/metrics/auc/ToFloat_2Castdnn/head/metrics/auc/LogicalAnd*

SrcT0
*(
_output_shapes
:╚         *

DstT0
П
dnn/head/metrics/auc/mulMuldnn/head/metrics/auc/ToFloat_2dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
l
*dnn/head/metrics/auc/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
и
dnn/head/metrics/auc/SumSumdnn/head/metrics/auc/mul*dnn/head/metrics/auc/Sum/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
█
dnn/head/metrics/auc/AssignAdd	AssignAdd#dnn/head/metrics/auc/true_positivesdnn/head/metrics/auc/Sum*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
_output_shapes	
:╚
╛
6dnn/head/metrics/auc/false_negatives/Initializer/zerosConst*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╦
$dnn/head/metrics/auc/false_negatives
VariableV2*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name 
Ы
+dnn/head/metrics/auc/false_negatives/AssignAssign$dnn/head/metrics/auc/false_negatives6dnn/head/metrics/auc/false_negatives/Initializer/zeros*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
validate_shape(*
_output_shapes	
:╚*
use_locking(
║
)dnn/head/metrics/auc/false_negatives/readIdentity$dnn/head/metrics/auc/false_negatives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:╚
Ч
!dnn/head/metrics/auc/LogicalAnd_1
LogicalAnddnn/head/metrics/auc/Tile_2dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
Л
dnn/head/metrics/auc/ToFloat_3Cast!dnn/head/metrics/auc/LogicalAnd_1*

SrcT0
*(
_output_shapes
:╚         *

DstT0
С
dnn/head/metrics/auc/mul_1Muldnn/head/metrics/auc/ToFloat_3dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
n
,dnn/head/metrics/auc/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
о
dnn/head/metrics/auc/Sum_1Sumdnn/head/metrics/auc/mul_1,dnn/head/metrics/auc/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
с
 dnn/head/metrics/auc/AssignAdd_1	AssignAdd$dnn/head/metrics/auc/false_negativesdnn/head/metrics/auc/Sum_1*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
_output_shapes	
:╚*
use_locking( 
╝
5dnn/head/metrics/auc/true_negatives/Initializer/zerosConst*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╔
#dnn/head/metrics/auc/true_negatives
VariableV2*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name 
Ч
*dnn/head/metrics/auc/true_negatives/AssignAssign#dnn/head/metrics/auc/true_negatives5dnn/head/metrics/auc/true_negatives/Initializer/zeros*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
validate_shape(*
_output_shapes	
:╚*
use_locking(
╖
(dnn/head/metrics/auc/true_negatives/readIdentity#dnn/head/metrics/auc/true_negatives*
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
_output_shapes	
:╚
Э
!dnn/head/metrics/auc/LogicalAnd_2
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/LogicalNot*(
_output_shapes
:╚         
Л
dnn/head/metrics/auc/ToFloat_4Cast!dnn/head/metrics/auc/LogicalAnd_2*

SrcT0
*(
_output_shapes
:╚         *

DstT0
С
dnn/head/metrics/auc/mul_2Muldnn/head/metrics/auc/ToFloat_4dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
n
,dnn/head/metrics/auc/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
о
dnn/head/metrics/auc/Sum_2Sumdnn/head/metrics/auc/mul_2,dnn/head/metrics/auc/Sum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
▀
 dnn/head/metrics/auc/AssignAdd_2	AssignAdd#dnn/head/metrics/auc/true_negativesdnn/head/metrics/auc/Sum_2*
use_locking( *
T0*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
_output_shapes	
:╚
╛
6dnn/head/metrics/auc/false_positives/Initializer/zerosConst*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
╦
$dnn/head/metrics/auc/false_positives
VariableV2*
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name *7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
	container 
Ы
+dnn/head/metrics/auc/false_positives/AssignAssign$dnn/head/metrics/auc/false_positives6dnn/head/metrics/auc/false_positives/Initializer/zeros*
use_locking(*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
validate_shape(*
_output_shapes	
:╚
║
)dnn/head/metrics/auc/false_positives/readIdentity$dnn/head/metrics/auc/false_positives*
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:╚
Ъ
!dnn/head/metrics/auc/LogicalAnd_3
LogicalAnd!dnn/head/metrics/auc/LogicalNot_1dnn/head/metrics/auc/Greater*(
_output_shapes
:╚         
Л
dnn/head/metrics/auc/ToFloat_5Cast!dnn/head/metrics/auc/LogicalAnd_3*

SrcT0
*(
_output_shapes
:╚         *

DstT0
С
dnn/head/metrics/auc/mul_3Muldnn/head/metrics/auc/ToFloat_5dnn/head/metrics/auc/Tile_3*
T0*(
_output_shapes
:╚         
n
,dnn/head/metrics/auc/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
о
dnn/head/metrics/auc/Sum_3Sumdnn/head/metrics/auc/mul_3,dnn/head/metrics/auc/Sum_3/reduction_indices*
T0*
_output_shapes	
:╚*

Tidx0*
	keep_dims( 
с
 dnn/head/metrics/auc/AssignAdd_3	AssignAdd$dnn/head/metrics/auc/false_positivesdnn/head/metrics/auc/Sum_3*
use_locking( *
T0*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
_output_shapes	
:╚
_
dnn/head/metrics/auc/add/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Л
dnn/head/metrics/auc/addAdd(dnn/head/metrics/auc/true_positives/readdnn/head/metrics/auc/add/y*
T0*
_output_shapes	
:╚
Ь
dnn/head/metrics/auc/add_1Add(dnn/head/metrics/auc/true_positives/read)dnn/head/metrics/auc/false_negatives/read*
T0*
_output_shapes	
:╚
a
dnn/head/metrics/auc/add_2/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Б
dnn/head/metrics/auc/add_2Adddnn/head/metrics/auc/add_1dnn/head/metrics/auc/add_2/y*
T0*
_output_shapes	
:╚

dnn/head/metrics/auc/divRealDivdnn/head/metrics/auc/adddnn/head/metrics/auc/add_2*
T0*
_output_shapes	
:╚
Ь
dnn/head/metrics/auc/add_3Add)dnn/head/metrics/auc/false_positives/read(dnn/head/metrics/auc/true_negatives/read*
T0*
_output_shapes	
:╚
a
dnn/head/metrics/auc/add_4/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Б
dnn/head/metrics/auc/add_4Adddnn/head/metrics/auc/add_3dnn/head/metrics/auc/add_4/y*
T0*
_output_shapes	
:╚
Т
dnn/head/metrics/auc/div_1RealDiv)dnn/head/metrics/auc/false_positives/readdnn/head/metrics/auc/add_4*
T0*
_output_shapes	
:╚
t
*dnn/head/metrics/auc/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_1/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_1StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_1/stack,dnn/head/metrics/auc/strided_slice_1/stack_1,dnn/head/metrics/auc/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
Index0*
T0
t
*dnn/head/metrics/auc/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_2StridedSlicednn/head/metrics/auc/div_1*dnn/head/metrics/auc/strided_slice_2/stack,dnn/head/metrics/auc/strided_slice_2/stack_1,dnn/head/metrics/auc/strided_slice_2/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟*
Index0*
T0
С
dnn/head/metrics/auc/subSub$dnn/head/metrics/auc/strided_slice_1$dnn/head/metrics/auc/strided_slice_2*
T0*
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_3/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
$dnn/head/metrics/auc/strided_slice_3StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_3/stack,dnn/head/metrics/auc/strided_slice_3/stack_1,dnn/head/metrics/auc/strided_slice_3/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
T0*
Index0*
shrink_axis_mask 
t
*dnn/head/metrics/auc/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
э
$dnn/head/metrics/auc/strided_slice_4StridedSlicednn/head/metrics/auc/div*dnn/head/metrics/auc/strided_slice_4/stack,dnn/head/metrics/auc/strided_slice_4/stack_1,dnn/head/metrics/auc/strided_slice_4/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟*
T0*
Index0
У
dnn/head/metrics/auc/add_5Add$dnn/head/metrics/auc/strided_slice_3$dnn/head/metrics/auc/strided_slice_4*
T0*
_output_shapes	
:╟
c
dnn/head/metrics/auc/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
Й
dnn/head/metrics/auc/truedivRealDivdnn/head/metrics/auc/add_5dnn/head/metrics/auc/truediv/y*
T0*
_output_shapes	
:╟
}
dnn/head/metrics/auc/MulMuldnn/head/metrics/auc/subdnn/head/metrics/auc/truediv*
T0*
_output_shapes	
:╟
f
dnn/head/metrics/auc/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ч
dnn/head/metrics/auc/valueSumdnn/head/metrics/auc/Muldnn/head/metrics/auc/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
a
dnn/head/metrics/auc/add_6/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Е
dnn/head/metrics/auc/add_6Adddnn/head/metrics/auc/AssignAdddnn/head/metrics/auc/add_6/y*
T0*
_output_shapes	
:╚
Й
dnn/head/metrics/auc/add_7Adddnn/head/metrics/auc/AssignAdd dnn/head/metrics/auc/AssignAdd_1*
T0*
_output_shapes	
:╚
a
dnn/head/metrics/auc/add_8/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Б
dnn/head/metrics/auc/add_8Adddnn/head/metrics/auc/add_7dnn/head/metrics/auc/add_8/y*
T0*
_output_shapes	
:╚
Г
dnn/head/metrics/auc/div_2RealDivdnn/head/metrics/auc/add_6dnn/head/metrics/auc/add_8*
T0*
_output_shapes	
:╚
Л
dnn/head/metrics/auc/add_9Add dnn/head/metrics/auc/AssignAdd_3 dnn/head/metrics/auc/AssignAdd_2*
T0*
_output_shapes	
:╚
b
dnn/head/metrics/auc/add_10/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
Г
dnn/head/metrics/auc/add_10Adddnn/head/metrics/auc/add_9dnn/head/metrics/auc/add_10/y*
T0*
_output_shapes	
:╚
К
dnn/head/metrics/auc/div_3RealDiv dnn/head/metrics/auc/AssignAdd_3dnn/head/metrics/auc/add_10*
T0*
_output_shapes	
:╚
t
*dnn/head/metrics/auc/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_5/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_5StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_5/stack,dnn/head/metrics/auc/strided_slice_5/stack_1,dnn/head/metrics/auc/strided_slice_5/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
T0*
Index0
t
*dnn/head/metrics/auc/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_6StridedSlicednn/head/metrics/auc/div_3*dnn/head/metrics/auc/strided_slice_6/stack,dnn/head/metrics/auc/strided_slice_6/stack_1,dnn/head/metrics/auc/strided_slice_6/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟*
T0*
Index0*
shrink_axis_mask 
У
dnn/head/metrics/auc/sub_1Sub$dnn/head/metrics/auc/strided_slice_5$dnn/head/metrics/auc/strided_slice_6*
T0*
_output_shapes	
:╟
t
*dnn/head/metrics/auc/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
w
,dnn/head/metrics/auc/strided_slice_7/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_7StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_7/stack,dnn/head/metrics/auc/strided_slice_7/stack_1,dnn/head/metrics/auc/strided_slice_7/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
Index0*
T0*
shrink_axis_mask 
t
*dnn/head/metrics/auc/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
v
,dnn/head/metrics/auc/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
я
$dnn/head/metrics/auc/strided_slice_8StridedSlicednn/head/metrics/auc/div_2*dnn/head/metrics/auc/strided_slice_8/stack,dnn/head/metrics/auc/strided_slice_8/stack_1,dnn/head/metrics/auc/strided_slice_8/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
Ф
dnn/head/metrics/auc/add_11Add$dnn/head/metrics/auc/strided_slice_7$dnn/head/metrics/auc/strided_slice_8*
T0*
_output_shapes	
:╟
e
 dnn/head/metrics/auc/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
О
dnn/head/metrics/auc/truediv_1RealDivdnn/head/metrics/auc/add_11 dnn/head/metrics/auc/truediv_1/y*
T0*
_output_shapes	
:╟
Г
dnn/head/metrics/auc/Mul_1Muldnn/head/metrics/auc/sub_1dnn/head/metrics/auc/truediv_1*
T0*
_output_shapes	
:╟
f
dnn/head/metrics/auc/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Э
dnn/head/metrics/auc/update_opSumdnn/head/metrics/auc/Mul_1dnn/head/metrics/auc/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
У
*dnn/head/metrics/auc_precision_recall/CastCastdnn/head/assert_range/Identity*

SrcT0*'
_output_shapes
:         *

DstT0

Д
?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
Zdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ы
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
╢
Ydnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Ъ
Xdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
p
hdnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
П
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeShapednn/head/predictions/logistici^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ў
Gdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ConstConsti^dnn/head/metrics/auc_precision_recall/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Н
Adnn/head/metrics/auc_precision_recall/broadcast_weights/ones_likeFillGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/ShapeGdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like/Const*
T0*'
_output_shapes
:         
Ї
7dnn/head/metrics/auc_precision_recall/broadcast_weightsMul?dnn/head/metrics/auc_precision_recall/broadcast_weights/weightsAdnn/head/metrics/auc_precision_recall/broadcast_weights/ones_like*
T0*'
_output_shapes
:         
s
.dnn/head/metrics/auc_precision_recall/Cast_1/xConst*
valueB
 *    *
dtype0*
_output_shapes
: 
╪
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqualGreaterEqualdnn/head/predictions/logistic.dnn/head/metrics/auc_precision_recall/Cast_1/x*
T0*'
_output_shapes
:         
С
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
Е
>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllAllGdnn/head/metrics/auc_precision_recall/assert_greater_equal/GreaterEqual@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
е
Gdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
█
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_1Const*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
┴
Idnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/Const_2Const*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
Б
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
T0
*
_output_shapes
: : 
█
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
┘
Vdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fIdentityTdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
┬
Udnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_idIdentity>dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
T0
*
_output_shapes
: 
│
Rdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOpNoOpW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t
е
`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependencyIdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_tS^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
Т
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╞
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x >= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
м
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3ConstW^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_1/x:0) = *
dtype0*
_output_shapes
: 
Є
[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/SwitchSwitch>dnn/head/metrics/auc_precision_recall/assert_greater_equal/AllUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0
*Q
_classG
ECloc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All*
_output_shapes
: : 
╘
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
╘
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2Switch.dnn/head/metrics/auc_precision_recall/Cast_1/xUdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id*
T0*A
_class7
53loc:@dnn/head/metrics/auc_precision_recall/Cast_1/x*
_output_shapes
: : 
▒
Tdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/AssertAssert[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1[dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
й
bdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1IdentityVdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_fU^dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*i
_class_
][loc:@dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
╬
Sdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/MergeMergebdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1`dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
s
.dnn/head/metrics/auc_precision_recall/Cast_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╧
Adnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual	LessEqualdnn/head/predictions/logistic.dnn/head/metrics/auc_precision_recall/Cast_2/x*
T0*'
_output_shapes
:         
О
=dnn/head/metrics/auc_precision_recall/assert_less_equal/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
∙
;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllAllAdnn/head/metrics/auc_precision_recall/assert_less_equal/LessEqual=dnn/head/metrics/auc_precision_recall/assert_less_equal/Const*
_output_shapes
: *
	keep_dims( *

Tidx0
в
Ddnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/ConstConst*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
╪
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_1Const*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
╛
Fdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/Const_2Const*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
°
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/All;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: : 
╒
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
╙
Sdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fIdentityQdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Switch*
T0
*
_output_shapes
: 
╝
Rdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_idIdentity;dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
T0
*
_output_shapes
: 
н
Odnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOpNoOpT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t
Щ
]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependencyIdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_tP^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t*
_output_shapes
: 
М
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*.
value%B# Bpredictions must be in [0, 1]*
dtype0*
_output_shapes
: 
└
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*b
valueYBW BQCondition x <= y did not hold element-wise:x (dnn/head/predictions/logistic:0) = *
dtype0*
_output_shapes
: 
ж
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3ConstT^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*H
value?B= B7y (dnn/head/metrics/auc_precision_recall/Cast_2/x:0) = *
dtype0*
_output_shapes
: 
ц
Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchSwitch;dnn/head/metrics/auc_precision_recall/assert_less_equal/AllRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0
*N
_classD
B@loc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/All*
_output_shapes
: : 
╬
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Switchdnn/head/predictions/logisticRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0*0
_class&
$"loc:@dnn/head/predictions/logistic*:
_output_shapes(
&:         :         
╬
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2Switch.dnn/head/metrics/auc_precision_recall/Cast_2/xRdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id*
T0*A
_class7
53loc:@dnn/head/metrics/auc_precision_recall/Cast_2/x*
_output_shapes
: : 
Ь
Qdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/AssertAssertXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/SwitchXdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1Xdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2*
T	
2*
	summarize
Э
_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1IdentitySdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_fR^dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert*
T0
*f
_class\
ZXloc:@dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f*
_output_shapes
: 
┼
Pdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/MergeMerge_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1]dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Д
3dnn/head/metrics/auc_precision_recall/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
╠
-dnn/head/metrics/auc_precision_recall/ReshapeReshapednn/head/predictions/logistic3dnn/head/metrics/auc_precision_recall/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Ж
5dnn/head/metrics/auc_precision_recall/Reshape_1/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
▌
/dnn/head/metrics/auc_precision_recall/Reshape_1Reshape*dnn/head/metrics/auc_precision_recall/Cast5dnn/head/metrics/auc_precision_recall/Reshape_1/shape*
T0
*
Tshape0*'
_output_shapes
:         
Ш
+dnn/head/metrics/auc_precision_recall/ShapeShape-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
out_type0*
_output_shapes
:
Г
9dnn/head/metrics/auc_precision_recall/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Е
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Е
;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
╖
3dnn/head/metrics/auc_precision_recall/strided_sliceStridedSlice+dnn/head/metrics/auc_precision_recall/Shape9dnn/head/metrics/auc_precision_recall/strided_slice/stack;dnn/head/metrics/auc_precision_recall/strided_slice/stack_1;dnn/head/metrics/auc_precision_recall/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
Ъ
+dnn/head/metrics/auc_precision_recall/ConstConst*╣
valueпBм╚"аХ┐╓│╧йд;╧й$<╖■v<╧йд<C╘═<╖■Ў<Х=╧й$=	?9=C╘M=}ib=╖■v=°╔Е=ХР=2_Ъ=╧йд=lЇо=	?╣=жЙ├=C╘═=р╪=}iт=┤ь=╖■Ў=кд >°╔>Gя
>Х>ф9>2_>БД>╧й$>╧)>lЇ.>╗4>	?9>Wd>>жЙC>ЇоH>C╘M>С∙R>рX>.D]>}ib>╦Оg>┤l>h┘q>╖■v>$|>кдА>Q7Г>°╔Е>а\И>GяК>юБН>ХР><зТ>ф9Х>Л╠Ч>2_Ъ>┘ёЬ>БДЯ>(в>╧йд>v<з>╧й>┼aм>lЇо>З▒>╗┤>bм╢>	?╣>░╤╗>Wd╛> Ў└>жЙ├>M╞>Їо╚>ЬA╦>C╘═>ъf╨>С∙╥>9М╒>р╪>З▒┌>.D▌>╓╓▀>}iт>$№ф>╦Оч>r!ъ>┤ь>┴Fя>h┘ё>lЇ>╖■Ў>^С∙>$№>м╢■>кд ?¤э?Q7?еА?°╔?L?а\?єе	?Gя
?Ъ8?юБ?B╦?Х?щ]?<з?РЁ?ф9?7Г?Л╠?▀?2_?Жи?┘ё?-;?БД?╘═ ?("?{`#?╧й$?#є%?v<'?╩Е(?╧)?q+?┼a,?л-?lЇ.?└=0?З1?g╨2?╗4?c5?bм6?╡ї7?	?9?]И:?░╤;?=?Wd>?лн?? Ў@?R@B?жЙC?·╥D?MF?бeG?ЇоH?H°I?ЬAK?яКL?C╘M?ЧO?ъfP?>░Q?С∙R?хBT?9МU?М╒V?рX?3hY?З▒Z?█·[?.D]?ВН^?╓╓_?) a?}ib?╨▓c?$№d?xEf?╦Оg?╪h?r!j?╞jk?┤l?m¤m?┴Fo?Рp?h┘q?╝"s?lt?c╡u?╖■v?
Hx?^Сy?▓┌z?$|?Ym}?м╢~? А?*
dtype0*
_output_shapes	
:╚
~
4dnn/head/metrics/auc_precision_recall/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
╫
0dnn/head/metrics/auc_precision_recall/ExpandDims
ExpandDims+dnn/head/metrics/auc_precision_recall/Const4dnn/head/metrics/auc_precision_recall/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	╚
o
-dnn/head/metrics/auc_precision_recall/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
╤
+dnn/head/metrics/auc_precision_recall/stackPack-dnn/head/metrics/auc_precision_recall/stack/03dnn/head/metrics/auc_precision_recall/strided_slice*
T0*

axis *
N*
_output_shapes
:
╓
*dnn/head/metrics/auc_precision_recall/TileTile0dnn/head/metrics/auc_precision_recall/ExpandDims+dnn/head/metrics/auc_precision_recall/stack*
T0*(
_output_shapes
:╚         *

Tmultiples0
М
4dnn/head/metrics/auc_precision_recall/transpose/RankRank-dnn/head/metrics/auc_precision_recall/Reshape*
T0*
_output_shapes
: 
w
5dnn/head/metrics/auc_precision_recall/transpose/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
╚
3dnn/head/metrics/auc_precision_recall/transpose/subSub4dnn/head/metrics/auc_precision_recall/transpose/Rank5dnn/head/metrics/auc_precision_recall/transpose/sub/y*
T0*
_output_shapes
: 
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/startConst*
value	B : *
dtype0*
_output_shapes
: 
}
;dnn/head/metrics/auc_precision_recall/transpose/Range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ц
5dnn/head/metrics/auc_precision_recall/transpose/RangeRange;dnn/head/metrics/auc_precision_recall/transpose/Range/start4dnn/head/metrics/auc_precision_recall/transpose/Rank;dnn/head/metrics/auc_precision_recall/transpose/Range/delta*
_output_shapes
:*

Tidx0
═
5dnn/head/metrics/auc_precision_recall/transpose/sub_1Sub3dnn/head/metrics/auc_precision_recall/transpose/sub5dnn/head/metrics/auc_precision_recall/transpose/Range*
T0*
_output_shapes
:
с
/dnn/head/metrics/auc_precision_recall/transpose	Transpose-dnn/head/metrics/auc_precision_recall/Reshape5dnn/head/metrics/auc_precision_recall/transpose/sub_1*
T0*'
_output_shapes
:         *
Tperm0
З
6dnn/head/metrics/auc_precision_recall/Tile_1/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
т
,dnn/head/metrics/auc_precision_recall/Tile_1Tile/dnn/head/metrics/auc_precision_recall/transpose6dnn/head/metrics/auc_precision_recall/Tile_1/multiples*

Tmultiples0*
T0*(
_output_shapes
:╚         
┼
-dnn/head/metrics/auc_precision_recall/GreaterGreater,dnn/head/metrics/auc_precision_recall/Tile_1*dnn/head/metrics/auc_precision_recall/Tile*
T0*(
_output_shapes
:╚         
Ч
0dnn/head/metrics/auc_precision_recall/LogicalNot
LogicalNot-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:╚         
З
6dnn/head/metrics/auc_precision_recall/Tile_2/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
т
,dnn/head/metrics/auc_precision_recall/Tile_2Tile/dnn/head/metrics/auc_precision_recall/Reshape_16dnn/head/metrics/auc_precision_recall/Tile_2/multiples*

Tmultiples0*
T0
*(
_output_shapes
:╚         
Ш
2dnn/head/metrics/auc_precision_recall/LogicalNot_1
LogicalNot,dnn/head/metrics/auc_precision_recall/Tile_2*(
_output_shapes
:╚         
╙
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeShape7dnn/head/metrics/auc_precision_recall/broadcast_weights*
T0*
out_type0*
_output_shapes
:
Э
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankConst*
value	B :*
dtype0*
_output_shapes
: 
╕
[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeShapednn/head/predictions/logistic*
T0*
out_type0*
_output_shapes
:
Ь
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
Ь
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
╗
Xdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarEqualZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar/x[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
┼
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/SwitchSwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: : 
√
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_tIdentityfdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
∙
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_fIdentityddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
ь
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_idIdentityXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
┴
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
ы
Лdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchZdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*m
_classc
a_loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank*
_output_shapes
: : 
я
Нdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rankednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank*
_output_shapes
: : 
═
Дdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualЛdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchНdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
╣
~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitchДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: : 
▒
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityАdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 
о
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentity~dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
│
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentityДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 
ч
Чdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
Д
Ъdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
р
Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1SwitchЪdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
Т
Уdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsЮdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1Чdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*
T0*
_output_shapes

:*

Tdim0
ю
Шdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB"      *
dtype0*
_output_shapes
:
▀
Шdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
·
Тdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillШdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeШdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*
_output_shapes

:
█
Фdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
Ь
Пdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2Уdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsТdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeФdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*
_output_shapes

:*

Tidx0
щ
Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
         *
dtype0*
_output_shapes
: 
И
Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapeednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
х
Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1SwitchЬdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
Ш
Хdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsаdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
T0*
_output_shapes

:*

Tdim0
▀
бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationХdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1Пdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
set_operationa-b*
T0*
validate_indices(*<
_output_shapes*
(:         :         :
∙
Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizeгdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
T0*
out_type0*
_output_shapes
: 
╤
Кdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstБ^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
▄
Иdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualКdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xЩdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
╙
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1SwitchДdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*Ъ
_classП
МЙloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
└
}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeАdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Иdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
Б
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/MergeMerge}dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Mergehdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
╝
Tdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/ConstConst*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
е
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╨
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_2Const*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
д
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
╢
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_4Const*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
б
Vdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╪
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergecdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
ї
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_tIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
є
cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fIdentityadnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ї
bdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_idIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
═
_dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOpNoOpd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t
┘
mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependencyIdentitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t`^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/NoOp*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
╢
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Э
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
╚
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*J
valueAB? B9dnn/head/metrics/auc_precision_recall/broadcast_weights:0*
dtype0*
_output_shapes
: 
Ь
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
о
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*0
value'B% Bdnn/head/predictions/logistic:0*
dtype0*
_output_shapes
: 
Щ
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7Constd^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
╓
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/SwitchSwitchcdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Mergebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
╥
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1Switch\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*o
_classe
caloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape* 
_output_shapes
::
╨
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2Switch[dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shapebdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0*n
_classd
b`loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape* 
_output_shapes
::
┬
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchXdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalarbdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id*
T0
*k
_classa
_]loc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar*
_output_shapes
: : 
║	
adnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/AssertAsserthdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switchhdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
▌
odnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1Identitycdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_fb^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert*
T0
*v
_classl
jhloc:@dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
ї
`dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/MergeMergeodnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1mdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 
Й
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeShapednn/head/predictions/logistica^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*
_output_shapes
:
ё
Idnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ConstConsta^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Merge*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
Cdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_likeFillIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/ShapeIdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like/Const*
T0*'
_output_shapes
:         
Ё
9dnn/head/metrics/auc_precision_recall/broadcast_weights_1Mul7dnn/head/metrics/auc_precision_recall/broadcast_weightsCdnn/head/metrics/auc_precision_recall/broadcast_weights_1/ones_like*
T0*'
_output_shapes
:         
Ж
5dnn/head/metrics/auc_precision_recall/Reshape_2/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
ь
/dnn/head/metrics/auc_precision_recall/Reshape_2Reshape9dnn/head/metrics/auc_precision_recall/broadcast_weights_15dnn/head/metrics/auc_precision_recall/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:         
З
6dnn/head/metrics/auc_precision_recall/Tile_3/multiplesConst*
valueB"╚      *
dtype0*
_output_shapes
:
т
,dnn/head/metrics/auc_precision_recall/Tile_3Tile/dnn/head/metrics/auc_precision_recall/Reshape_26dnn/head/metrics/auc_precision_recall/Tile_3/multiples*
T0*(
_output_shapes
:╚         *

Tmultiples0
▐
Fdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
ы
4dnn/head/metrics/auc_precision_recall/true_positives
VariableV2*
dtype0*
_output_shapes	
:╚*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
	container *
shape:╚
█
;dnn/head/metrics/auc_precision_recall/true_positives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_positivesFdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
validate_shape(*
_output_shapes	
:╚
ъ
9dnn/head/metrics/auc_precision_recall/true_positives/readIdentity4dnn/head/metrics/auc_precision_recall/true_positives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:╚
┼
0dnn/head/metrics/auc_precision_recall/LogicalAnd
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_2-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:╚         
л
/dnn/head/metrics/auc_precision_recall/ToFloat_2Cast0dnn/head/metrics/auc_precision_recall/LogicalAnd*

SrcT0
*(
_output_shapes
:╚         *

DstT0
┬
)dnn/head/metrics/auc_precision_recall/mulMul/dnn/head/metrics/auc_precision_recall/ToFloat_2,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         
}
;dnn/head/metrics/auc_precision_recall/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
█
)dnn/head/metrics/auc_precision_recall/SumSum)dnn/head/metrics/auc_precision_recall/mul;dnn/head/metrics/auc_precision_recall/Sum/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
Я
/dnn/head/metrics/auc_precision_recall/AssignAdd	AssignAdd4dnn/head/metrics/auc_precision_recall/true_positives)dnn/head/metrics/auc_precision_recall/Sum*
use_locking( *
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
_output_shapes	
:╚
р
Gdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
э
5dnn/head/metrics/auc_precision_recall/false_negatives
VariableV2*
shared_name *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
	container *
shape:╚*
dtype0*
_output_shapes	
:╚
▀
<dnn/head/metrics/auc_precision_recall/false_negatives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_negativesGdnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zeros*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
validate_shape(*
_output_shapes	
:╚*
use_locking(
э
:dnn/head/metrics/auc_precision_recall/false_negatives/readIdentity5dnn/head/metrics/auc_precision_recall/false_negatives*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:╚
╩
2dnn/head/metrics/auc_precision_recall/LogicalAnd_1
LogicalAnd,dnn/head/metrics/auc_precision_recall/Tile_20dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:╚         
н
/dnn/head/metrics/auc_precision_recall/ToFloat_3Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_1*

SrcT0
*(
_output_shapes
:╚         *

DstT0
─
+dnn/head/metrics/auc_precision_recall/mul_1Mul/dnn/head/metrics/auc_precision_recall/ToFloat_3,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         

=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
+dnn/head/metrics/auc_precision_recall/Sum_1Sum+dnn/head/metrics/auc_precision_recall/mul_1=dnn/head/metrics/auc_precision_recall/Sum_1/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
е
1dnn/head/metrics/auc_precision_recall/AssignAdd_1	AssignAdd5dnn/head/metrics/auc_precision_recall/false_negatives+dnn/head/metrics/auc_precision_recall/Sum_1*
use_locking( *
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
_output_shapes	
:╚
▐
Fdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zerosConst*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
ы
4dnn/head/metrics/auc_precision_recall/true_negatives
VariableV2*
dtype0*
_output_shapes	
:╚*
shared_name *G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
	container *
shape:╚
█
;dnn/head/metrics/auc_precision_recall/true_negatives/AssignAssign4dnn/head/metrics/auc_precision_recall/true_negativesFdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zeros*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
validate_shape(*
_output_shapes	
:╚*
use_locking(
ъ
9dnn/head/metrics/auc_precision_recall/true_negatives/readIdentity4dnn/head/metrics/auc_precision_recall/true_negatives*
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:╚
╨
2dnn/head/metrics/auc_precision_recall/LogicalAnd_2
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_10dnn/head/metrics/auc_precision_recall/LogicalNot*(
_output_shapes
:╚         
н
/dnn/head/metrics/auc_precision_recall/ToFloat_4Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_2*

SrcT0
*(
_output_shapes
:╚         *

DstT0
─
+dnn/head/metrics/auc_precision_recall/mul_2Mul/dnn/head/metrics/auc_precision_recall/ToFloat_4,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         

=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
+dnn/head/metrics/auc_precision_recall/Sum_2Sum+dnn/head/metrics/auc_precision_recall/mul_2=dnn/head/metrics/auc_precision_recall/Sum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
г
1dnn/head/metrics/auc_precision_recall/AssignAdd_2	AssignAdd4dnn/head/metrics/auc_precision_recall/true_negatives+dnn/head/metrics/auc_precision_recall/Sum_2*
use_locking( *
T0*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
_output_shapes	
:╚
р
Gdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zerosConst*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
valueB╚*    *
dtype0*
_output_shapes	
:╚
э
5dnn/head/metrics/auc_precision_recall/false_positives
VariableV2*
shape:╚*
dtype0*
_output_shapes	
:╚*
shared_name *H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
	container 
▀
<dnn/head/metrics/auc_precision_recall/false_positives/AssignAssign5dnn/head/metrics/auc_precision_recall/false_positivesGdnn/head/metrics/auc_precision_recall/false_positives/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
validate_shape(*
_output_shapes	
:╚
э
:dnn/head/metrics/auc_precision_recall/false_positives/readIdentity5dnn/head/metrics/auc_precision_recall/false_positives*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:╚
═
2dnn/head/metrics/auc_precision_recall/LogicalAnd_3
LogicalAnd2dnn/head/metrics/auc_precision_recall/LogicalNot_1-dnn/head/metrics/auc_precision_recall/Greater*(
_output_shapes
:╚         
н
/dnn/head/metrics/auc_precision_recall/ToFloat_5Cast2dnn/head/metrics/auc_precision_recall/LogicalAnd_3*

SrcT0
*(
_output_shapes
:╚         *

DstT0
─
+dnn/head/metrics/auc_precision_recall/mul_3Mul/dnn/head/metrics/auc_precision_recall/ToFloat_5,dnn/head/metrics/auc_precision_recall/Tile_3*
T0*(
_output_shapes
:╚         

=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
с
+dnn/head/metrics/auc_precision_recall/Sum_3Sum+dnn/head/metrics/auc_precision_recall/mul_3=dnn/head/metrics/auc_precision_recall/Sum_3/reduction_indices*

Tidx0*
	keep_dims( *
T0*
_output_shapes	
:╚
е
1dnn/head/metrics/auc_precision_recall/AssignAdd_3	AssignAdd5dnn/head/metrics/auc_precision_recall/false_positives+dnn/head/metrics/auc_precision_recall/Sum_3*
T0*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
_output_shapes	
:╚*
use_locking( 
p
+dnn/head/metrics/auc_precision_recall/add/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
╛
)dnn/head/metrics/auc_precision_recall/addAdd9dnn/head/metrics/auc_precision_recall/true_positives/read+dnn/head/metrics/auc_precision_recall/add/y*
T0*
_output_shapes	
:╚
╧
+dnn/head/metrics/auc_precision_recall/add_1Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_negatives/read*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_2/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┤
+dnn/head/metrics/auc_precision_recall/add_2Add+dnn/head/metrics/auc_precision_recall/add_1-dnn/head/metrics/auc_precision_recall/add_2/y*
T0*
_output_shapes	
:╚
▓
)dnn/head/metrics/auc_precision_recall/divRealDiv)dnn/head/metrics/auc_precision_recall/add+dnn/head/metrics/auc_precision_recall/add_2*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_3/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┬
+dnn/head/metrics/auc_precision_recall/add_3Add9dnn/head/metrics/auc_precision_recall/true_positives/read-dnn/head/metrics/auc_precision_recall/add_3/y*
T0*
_output_shapes	
:╚
╧
+dnn/head/metrics/auc_precision_recall/add_4Add9dnn/head/metrics/auc_precision_recall/true_positives/read:dnn/head/metrics/auc_precision_recall/false_positives/read*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_5/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┤
+dnn/head/metrics/auc_precision_recall/add_5Add+dnn/head/metrics/auc_precision_recall/add_4-dnn/head/metrics/auc_precision_recall/add_5/y*
T0*
_output_shapes	
:╚
╢
+dnn/head/metrics/auc_precision_recall/div_1RealDiv+dnn/head/metrics/auc_precision_recall/add_3+dnn/head/metrics/auc_precision_recall/add_5*
T0*
_output_shapes	
:╚
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┬
5dnn/head/metrics/auc_precision_recall/strided_slice_1StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_1/stack=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟*
Index0*
T0*
shrink_axis_mask 
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
┬
5dnn/head/metrics/auc_precision_recall/strided_slice_2StridedSlice)dnn/head/metrics/auc_precision_recall/div;dnn/head/metrics/auc_precision_recall/strided_slice_2/stack=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
─
)dnn/head/metrics/auc_precision_recall/subSub5dnn/head/metrics/auc_precision_recall/strided_slice_15dnn/head/metrics/auc_precision_recall/strided_slice_2*
T0*
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_3StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_3/stack=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_4StridedSlice+dnn/head/metrics/auc_precision_recall/div_1;dnn/head/metrics/auc_precision_recall/strided_slice_4/stack=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
╞
+dnn/head/metrics/auc_precision_recall/add_6Add5dnn/head/metrics/auc_precision_recall/strided_slice_35dnn/head/metrics/auc_precision_recall/strided_slice_4*
T0*
_output_shapes	
:╟
t
/dnn/head/metrics/auc_precision_recall/truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
╝
-dnn/head/metrics/auc_precision_recall/truedivRealDiv+dnn/head/metrics/auc_precision_recall/add_6/dnn/head/metrics/auc_precision_recall/truediv/y*
T0*
_output_shapes	
:╟
░
)dnn/head/metrics/auc_precision_recall/MulMul)dnn/head/metrics/auc_precision_recall/sub-dnn/head/metrics/auc_precision_recall/truediv*
T0*
_output_shapes	
:╟
w
-dnn/head/metrics/auc_precision_recall/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
╩
+dnn/head/metrics/auc_precision_recall/valueSum)dnn/head/metrics/auc_precision_recall/Mul-dnn/head/metrics/auc_precision_recall/Const_1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
r
-dnn/head/metrics/auc_precision_recall/add_7/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
╕
+dnn/head/metrics/auc_precision_recall/add_7Add/dnn/head/metrics/auc_precision_recall/AssignAdd-dnn/head/metrics/auc_precision_recall/add_7/y*
T0*
_output_shapes	
:╚
╝
+dnn/head/metrics/auc_precision_recall/add_8Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_1*
T0*
_output_shapes	
:╚
r
-dnn/head/metrics/auc_precision_recall/add_9/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
┤
+dnn/head/metrics/auc_precision_recall/add_9Add+dnn/head/metrics/auc_precision_recall/add_8-dnn/head/metrics/auc_precision_recall/add_9/y*
T0*
_output_shapes	
:╚
╢
+dnn/head/metrics/auc_precision_recall/div_2RealDiv+dnn/head/metrics/auc_precision_recall/add_7+dnn/head/metrics/auc_precision_recall/add_9*
T0*
_output_shapes	
:╚
s
.dnn/head/metrics/auc_precision_recall/add_10/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
║
,dnn/head/metrics/auc_precision_recall/add_10Add/dnn/head/metrics/auc_precision_recall/AssignAdd.dnn/head/metrics/auc_precision_recall/add_10/y*
T0*
_output_shapes	
:╚
╜
,dnn/head/metrics/auc_precision_recall/add_11Add/dnn/head/metrics/auc_precision_recall/AssignAdd1dnn/head/metrics/auc_precision_recall/AssignAdd_3*
T0*
_output_shapes	
:╚
s
.dnn/head/metrics/auc_precision_recall/add_12/yConst*
valueB
 *╜7Ж5*
dtype0*
_output_shapes
: 
╖
,dnn/head/metrics/auc_precision_recall/add_12Add,dnn/head/metrics/auc_precision_recall/add_11.dnn/head/metrics/auc_precision_recall/add_12/y*
T0*
_output_shapes	
:╚
╕
+dnn/head/metrics/auc_precision_recall/div_3RealDiv,dnn/head/metrics/auc_precision_recall/add_10,dnn/head/metrics/auc_precision_recall/add_12*
T0*
_output_shapes	
:╚
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_5/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_5StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_5/stack=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_6/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_6StridedSlice+dnn/head/metrics/auc_precision_recall/div_2;dnn/head/metrics/auc_precision_recall/strided_slice_6/stack=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
╞
+dnn/head/metrics/auc_precision_recall/sub_1Sub5dnn/head/metrics/auc_precision_recall/strided_slice_55dnn/head/metrics/auc_precision_recall/strided_slice_6*
T0*
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_7/stackConst*
valueB: *
dtype0*
_output_shapes
:
И
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1Const*
valueB:╟*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_7StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_7/stack=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes	
:╟
Е
;dnn/head/metrics/auc_precision_recall/strided_slice_8/stackConst*
valueB:*
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
З
=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
─
5dnn/head/metrics/auc_precision_recall/strided_slice_8StridedSlice+dnn/head/metrics/auc_precision_recall/div_3;dnn/head/metrics/auc_precision_recall/strided_slice_8/stack=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_1=dnn/head/metrics/auc_precision_recall/strided_slice_8/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes	
:╟
╟
,dnn/head/metrics/auc_precision_recall/add_13Add5dnn/head/metrics/auc_precision_recall/strided_slice_75dnn/head/metrics/auc_precision_recall/strided_slice_8*
T0*
_output_shapes	
:╟
v
1dnn/head/metrics/auc_precision_recall/truediv_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
┴
/dnn/head/metrics/auc_precision_recall/truediv_1RealDiv,dnn/head/metrics/auc_precision_recall/add_131dnn/head/metrics/auc_precision_recall/truediv_1/y*
T0*
_output_shapes	
:╟
╢
+dnn/head/metrics/auc_precision_recall/Mul_1Mul+dnn/head/metrics/auc_precision_recall/sub_1/dnn/head/metrics/auc_precision_recall/truediv_1*
T0*
_output_shapes	
:╟
w
-dnn/head/metrics/auc_precision_recall/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
╨
/dnn/head/metrics/auc_precision_recall/update_opSum+dnn/head/metrics/auc_precision_recall/Mul_1-dnn/head/metrics/auc_precision_recall/Const_2*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
А
mean/total/Initializer/zerosConst*
_class
loc:@mean/total*
valueB
 *    *
dtype0*
_output_shapes
: 
Н

mean/total
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@mean/total*
	container 
о
mean/total/AssignAssign
mean/totalmean/total/Initializer/zeros*
T0*
_class
loc:@mean/total*
validate_shape(*
_output_shapes
: *
use_locking(
g
mean/total/readIdentity
mean/total*
T0*
_class
loc:@mean/total*
_output_shapes
: 
А
mean/count/Initializer/zerosConst*
_class
loc:@mean/count*
valueB
 *    *
dtype0*
_output_shapes
: 
Н

mean/count
VariableV2*
_class
loc:@mean/count*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
о
mean/count/AssignAssign
mean/countmean/count/Initializer/zeros*
T0*
_class
loc:@mean/count*
validate_shape(*
_output_shapes
: *
use_locking(
g
mean/count/readIdentity
mean/count*
T0*
_class
loc:@mean/count*
_output_shapes
: 
K
	mean/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
Q
mean/ToFloat_1Cast	mean/Size*

SrcT0*
_output_shapes
: *

DstT0
M

mean/ConstConst*
valueB *
dtype0*
_output_shapes
: 
u
mean/SumSumdnn/head/weighted_loss/Sum
mean/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
Д
mean/AssignAdd	AssignAdd
mean/totalmean/Sum*
use_locking( *
T0*
_class
loc:@mean/total*
_output_shapes
: 
й
mean/AssignAdd_1	AssignAdd
mean/countmean/ToFloat_1^dnn/head/weighted_loss/Sum*
T0*
_class
loc:@mean/count*
_output_shapes
: *
use_locking( 
S
mean/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
mean/GreaterGreatermean/count/readmean/Greater/y*
T0*
_output_shapes
: 
Z
mean/truedivRealDivmean/total/readmean/count/read*
T0*
_output_shapes
: 
Q
mean/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_

mean/valueSelectmean/Greatermean/truedivmean/value/e*
T0*
_output_shapes
: 
U
mean/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
mean/Greater_1Greatermean/AssignAdd_1mean/Greater_1/y*
T0*
_output_shapes
: 
\
mean/truediv_1RealDivmean/AssignAddmean/AssignAdd_1*
T0*
_output_shapes
: 
U
mean/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
mean/update_opSelectmean/Greater_1mean/truediv_1mean/update_op/e*
T0*
_output_shapes
: 
╩

group_depsNoOp$^dnn/head/metrics/accuracy/update_op-^dnn/head/metrics/accuracy_baseline/update_op^dnn/head/metrics/auc/update_op0^dnn/head/metrics/auc_precision_recall/update_op(^dnn/head/metrics/average_loss/update_op&^dnn/head/metrics/label/mean/update_op^mean/update_op+^dnn/head/metrics/prediction/mean/update_op
{
eval_step/Initializer/zerosConst*
_class
loc:@eval_step*
value	B	 R *
dtype0	*
_output_shapes
: 
Л
	eval_step
VariableV2*
shape: *
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@eval_step*
	container 
к
eval_step/AssignAssign	eval_stepeval_step/Initializer/zeros*
T0	*
_class
loc:@eval_step*
validate_shape(*
_output_shapes
: *
use_locking(
d
eval_step/readIdentity	eval_step*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
Q
AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Д
	AssignAdd	AssignAdd	eval_stepAssignAdd/value*
use_locking(*
T0	*
_class
loc:@eval_step*
_output_shapes
: 
U
readIdentity	eval_step^group_deps
^AssignAdd*
T0	*
_output_shapes
: 
;
IdentityIdentityread*
T0	*
_output_shapes
: 
С
initNoOp^global_step/Assign`^dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/AssignZ^dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Assign'^dnn/hiddenlayer_0/kernel/part_0/Assign%^dnn/hiddenlayer_0/bias/part_0/Assign'^dnn/hiddenlayer_1/kernel/part_0/Assign%^dnn/hiddenlayer_1/bias/part_0/Assign'^dnn/hiddenlayer_2/kernel/part_0/Assign%^dnn/hiddenlayer_2/bias/part_0/Assign ^dnn/logits/kernel/part_0/Assign^dnn/logits/bias/part_0/Assign

init_1NoOp
$
group_deps_1NoOp^init^init_1
Я
4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
п
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
┼
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
┼
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
╔
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
┼
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_2/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
: 
╗
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
╕
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
╬
7report_uninitialized_variables/IsVariableInitialized_11IsVariableInitialized!dnn/head/metrics/label/mean/total*4
_class*
(&loc:@dnn/head/metrics/label/mean/total*
dtype0*
_output_shapes
: 
╬
7report_uninitialized_variables/IsVariableInitialized_12IsVariableInitialized!dnn/head/metrics/label/mean/count*4
_class*
(&loc:@dnn/head/metrics/label/mean/count*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_13IsVariableInitialized#dnn/head/metrics/average_loss/total*6
_class,
*(loc:@dnn/head/metrics/average_loss/total*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_14IsVariableInitialized#dnn/head/metrics/average_loss/count*6
_class,
*(loc:@dnn/head/metrics/average_loss/count*
dtype0*
_output_shapes
: 
╩
7report_uninitialized_variables/IsVariableInitialized_15IsVariableInitializeddnn/head/metrics/accuracy/total*2
_class(
&$loc:@dnn/head/metrics/accuracy/total*
dtype0*
_output_shapes
: 
╩
7report_uninitialized_variables/IsVariableInitialized_16IsVariableInitializeddnn/head/metrics/accuracy/count*2
_class(
&$loc:@dnn/head/metrics/accuracy/count*
dtype0*
_output_shapes
: 
╪
7report_uninitialized_variables/IsVariableInitialized_17IsVariableInitialized&dnn/head/metrics/prediction/mean/total*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/total*
dtype0*
_output_shapes
: 
╪
7report_uninitialized_variables/IsVariableInitialized_18IsVariableInitialized&dnn/head/metrics/prediction/mean/count*9
_class/
-+loc:@dnn/head/metrics/prediction/mean/count*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_19IsVariableInitialized#dnn/head/metrics/auc/true_positives*6
_class,
*(loc:@dnn/head/metrics/auc/true_positives*
dtype0*
_output_shapes
: 
╘
7report_uninitialized_variables/IsVariableInitialized_20IsVariableInitialized$dnn/head/metrics/auc/false_negatives*7
_class-
+)loc:@dnn/head/metrics/auc/false_negatives*
dtype0*
_output_shapes
: 
╥
7report_uninitialized_variables/IsVariableInitialized_21IsVariableInitialized#dnn/head/metrics/auc/true_negatives*6
_class,
*(loc:@dnn/head/metrics/auc/true_negatives*
dtype0*
_output_shapes
: 
╘
7report_uninitialized_variables/IsVariableInitialized_22IsVariableInitialized$dnn/head/metrics/auc/false_positives*7
_class-
+)loc:@dnn/head/metrics/auc/false_positives*
dtype0*
_output_shapes
: 
Ї
7report_uninitialized_variables/IsVariableInitialized_23IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_positives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_positives*
dtype0*
_output_shapes
: 
Ў
7report_uninitialized_variables/IsVariableInitialized_24IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_negatives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_negatives*
dtype0*
_output_shapes
: 
Ї
7report_uninitialized_variables/IsVariableInitialized_25IsVariableInitialized4dnn/head/metrics/auc_precision_recall/true_negatives*G
_class=
;9loc:@dnn/head/metrics/auc_precision_recall/true_negatives*
dtype0*
_output_shapes
: 
Ў
7report_uninitialized_variables/IsVariableInitialized_26IsVariableInitialized5dnn/head/metrics/auc_precision_recall/false_positives*H
_class>
<:loc:@dnn/head/metrics/auc_precision_recall/false_positives*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_27IsVariableInitialized
mean/total*
_class
loc:@mean/total*
dtype0*
_output_shapes
: 
а
7report_uninitialized_variables/IsVariableInitialized_28IsVariableInitialized
mean/count*
_class
loc:@mean/count*
dtype0*
_output_shapes
: 
Ю
7report_uninitialized_variables/IsVariableInitialized_29IsVariableInitialized	eval_step*
_class
loc:@eval_step*
dtype0	*
_output_shapes
: 
Ч
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_107report_uninitialized_variables/IsVariableInitialized_117report_uninitialized_variables/IsVariableInitialized_127report_uninitialized_variables/IsVariableInitialized_137report_uninitialized_variables/IsVariableInitialized_147report_uninitialized_variables/IsVariableInitialized_157report_uninitialized_variables/IsVariableInitialized_167report_uninitialized_variables/IsVariableInitialized_177report_uninitialized_variables/IsVariableInitialized_187report_uninitialized_variables/IsVariableInitialized_197report_uninitialized_variables/IsVariableInitialized_207report_uninitialized_variables/IsVariableInitialized_217report_uninitialized_variables/IsVariableInitialized_227report_uninitialized_variables/IsVariableInitialized_237report_uninitialized_variables/IsVariableInitialized_247report_uninitialized_variables/IsVariableInitialized_257report_uninitialized_variables/IsVariableInitialized_267report_uninitialized_variables/IsVariableInitialized_277report_uninitialized_variables/IsVariableInitialized_287report_uninitialized_variables/IsVariableInitialized_29"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
И
)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
▀	
$report_uninitialized_variables/ConstConst"/device:CPU:0*ў
valueэBъBglobal_stepBXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0BRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0Bdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/hiddenlayer_2/kernel/part_0Bdnn/hiddenlayer_2/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0B!dnn/head/metrics/label/mean/totalB!dnn/head/metrics/label/mean/countB#dnn/head/metrics/average_loss/totalB#dnn/head/metrics/average_loss/countBdnn/head/metrics/accuracy/totalBdnn/head/metrics/accuracy/countB&dnn/head/metrics/prediction/mean/totalB&dnn/head/metrics/prediction/mean/countB#dnn/head/metrics/auc/true_positivesB$dnn/head/metrics/auc/false_negativesB#dnn/head/metrics/auc/true_negativesB$dnn/head/metrics/auc/false_positivesB4dnn/head/metrics/auc_precision_recall/true_positivesB5dnn/head/metrics/auc_precision_recall/false_negativesB4dnn/head/metrics/auc_precision_recall/true_negativesB5dnn/head/metrics/auc_precision_recall/false_positivesB
mean/totalB
mean/countB	eval_step*
dtype0*
_output_shapes
:
К
1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ш
?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Ы
Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Д
0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
М
3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ё
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: 
╛
;report_uninitialized_variables/boolean_mask/concat/values_0Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
И
7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
║
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/concat/values_0;report_uninitialized_variables/boolean_mask/strided_slice_17report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
┌
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
Э
;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
         *
dtype0*
_output_shapes
:
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
й
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:         
┼
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*
T0	*#
_output_shapes
:         *
squeeze_dims

С
2report_uninitialized_variables/boolean_mask/GatherGather3report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze"/device:CPU:0*
Tindices0	*
Tparams0*
validate_indices(*#
_output_shapes
:         
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╝
concatConcatV22report_uninitialized_variables/boolean_mask/Gather$report_uninitialized_resources/Constconcat/axis*
T0*
N*#
_output_shapes
:         *

Tidx0
б
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
▒
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializeddnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
╟
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializeddnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializeddnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
╟
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializeddnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
╦
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializeddnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
╟
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializeddnn/hiddenlayer_2/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
: 
╜
8report_uninitialized_variables_1/IsVariableInitialized_9IsVariableInitializeddnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
║
9report_uninitialized_variables_1/IsVariableInitialized_10IsVariableInitializeddnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
Ї
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_88report_uninitialized_variables_1/IsVariableInitialized_99report_uninitialized_variables_1/IsVariableInitialized_10"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:
М
+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:
м
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*┬
value╕B╡Bglobal_stepBXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0BRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0Bdnn/hiddenlayer_0/kernel/part_0Bdnn/hiddenlayer_0/bias/part_0Bdnn/hiddenlayer_1/kernel/part_0Bdnn/hiddenlayer_1/bias/part_0Bdnn/hiddenlayer_2/kernel/part_0Bdnn/hiddenlayer_2/bias/part_0Bdnn/logits/kernel/part_0Bdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
М
3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ъ
Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Є
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
Э
Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
К
2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
О
5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ь
Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
Ю
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
Ю
Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
·
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask 
┬
=report_uninitialized_variables_1/boolean_mask/concat/values_0Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
К
9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
┬
4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/concat/values_0=report_uninitialized_variables_1/boolean_mask/strided_slice_19report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*

Tidx0*
T0*
N*
_output_shapes
:
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:
Я
=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
         *
dtype0*
_output_shapes
:
Ё
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
н
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:         
╔
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*
squeeze_dims
*
T0	*#
_output_shapes
:         
Ч
4report_uninitialized_variables_1/boolean_mask/GatherGather5report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze"/device:CPU:0*
Tparams0*
validate_indices(*#
_output_shapes
:         *
Tindices0	
█
init_2NoOp)^dnn/head/metrics/label/mean/total/Assign)^dnn/head/metrics/label/mean/count/Assign+^dnn/head/metrics/average_loss/total/Assign+^dnn/head/metrics/average_loss/count/Assign'^dnn/head/metrics/accuracy/total/Assign'^dnn/head/metrics/accuracy/count/Assign.^dnn/head/metrics/prediction/mean/total/Assign.^dnn/head/metrics/prediction/mean/count/Assign+^dnn/head/metrics/auc/true_positives/Assign,^dnn/head/metrics/auc/false_negatives/Assign+^dnn/head/metrics/auc/true_negatives/Assign,^dnn/head/metrics/auc/false_positives/Assign<^dnn/head/metrics/auc_precision_recall/true_positives/Assign=^dnn/head/metrics/auc_precision_recall/false_negatives/Assign<^dnn/head/metrics/auc_precision_recall/true_negatives/Assign=^dnn/head/metrics/auc_precision_recall/false_positives/Assign^mean/total/Assign^mean/count/Assign^eval_step/Assign
▀
init_all_tablesNoOph^dnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init\^dnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init
/
group_deps_2NoOp^init_2^init_all_tables
∙
Merge/MergeSummaryMergeSummary-dnn/dnn/hiddenlayer_0/fraction_of_zero_values dnn/dnn/hiddenlayer_0/activation-dnn/dnn/hiddenlayer_1/fraction_of_zero_values dnn/dnn/hiddenlayer_1/activation-dnn/dnn/hiddenlayer_2/fraction_of_zero_values dnn/dnn/hiddenlayer_2/activation&dnn/dnn/logits/fraction_of_zero_valuesdnn/dnn/logits/activation*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_37663ef8eafb4953b0fd37844185b92f/part*
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
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
╔
save/SaveV2/tensor_namesConst*№
valueЄBяBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBQdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weightsBKdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weightsBdnn/logits/biasBdnn/logits/kernelBglobal_step*
dtype0*
_output_shapes
:
Ї
save/SaveV2/shape_and_slicesConst*г
valueЩBЦB50 0,50B57 50 0,57:0,50B	100 0,100B50 100 0,50:0,100B50 0,50B100 50 0,100:0,50B12 12 0,12:0,12B32 32 0,32:0,32B1 0,1B50 1 0,50:0,1B *
dtype0*
_output_shapes
:
▌
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices"dnn/hiddenlayer_0/bias/part_0/read$dnn/hiddenlayer_0/kernel/part_0/read"dnn/hiddenlayer_1/bias/part_0/read$dnn/hiddenlayer_1/kernel/part_0/read"dnn/hiddenlayer_2/bias/part_0/read$dnn/hiddenlayer_2/kernel/part_0/read]dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/readWdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/readdnn/logits/bias/part_0/readdnn/logits/kernel/part_0/readglobal_step*
dtypes
2	
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
z
save/RestoreV2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_0/bias*
dtype0*
_output_shapes
:
o
save/RestoreV2/shape_and_slicesConst*
valueBB50 0,50*
dtype0*
_output_shapes
:
Т
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:2
─
save/AssignAssigndnn/hiddenlayer_0/bias/part_0save/RestoreV2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
validate_shape(*
_output_shapes
:2*
use_locking(
~
save/RestoreV2_1/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_0/kernel*
dtype0*
_output_shapes
:
y
!save/RestoreV2_1/shape_and_slicesConst*$
valueBB57 50 0,57:0,50*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes

:92
╨
save/Assign_1Assigndnn/hiddenlayer_0/kernel/part_0save/RestoreV2_1*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
validate_shape(*
_output_shapes

:92
|
save/RestoreV2_2/tensor_namesConst*+
value"B Bdnn/hiddenlayer_1/bias*
dtype0*
_output_shapes
:
s
!save/RestoreV2_2/shape_and_slicesConst*
valueBB	100 0,100*
dtype0*
_output_shapes
:
Ш
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:d*
dtypes
2
╚
save/Assign_2Assigndnn/hiddenlayer_1/bias/part_0save/RestoreV2_2*
T0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
validate_shape(*
_output_shapes
:d*
use_locking(
~
save/RestoreV2_3/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_1/kernel*
dtype0*
_output_shapes
:
{
!save/RestoreV2_3/shape_and_slicesConst*&
valueBB50 100 0,50:0,100*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
_output_shapes

:2d*
dtypes
2
╨
save/Assign_3Assigndnn/hiddenlayer_1/kernel/part_0save/RestoreV2_3*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
validate_shape(*
_output_shapes

:2d
|
save/RestoreV2_4/tensor_namesConst*+
value"B Bdnn/hiddenlayer_2/bias*
dtype0*
_output_shapes
:
q
!save/RestoreV2_4/shape_and_slicesConst*
valueBB50 0,50*
dtype0*
_output_shapes
:
Ш
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:2*
dtypes
2
╚
save/Assign_4Assigndnn/hiddenlayer_2/bias/part_0save/RestoreV2_4*
T0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
validate_shape(*
_output_shapes
:2*
use_locking(
~
save/RestoreV2_5/tensor_namesConst*-
value$B"Bdnn/hiddenlayer_2/kernel*
dtype0*
_output_shapes
:
{
!save/RestoreV2_5/shape_and_slicesConst*&
valueBB100 50 0,100:0,50*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes

:d2*
dtypes
2
╨
save/Assign_5Assigndnn/hiddenlayer_2/kernel/part_0save/RestoreV2_5*
use_locking(*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
validate_shape(*
_output_shapes

:d2
╖
save/RestoreV2_6/tensor_namesConst*f
value]B[BQdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights*
dtype0*
_output_shapes
:
y
!save/RestoreV2_6/shape_and_slicesConst*$
valueBB12 12 0,12:0,12*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes

:*
dtypes
2
┬
save/Assign_6AssignXdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0save/RestoreV2_6*
use_locking(*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:
▒
save/RestoreV2_7/tensor_namesConst*`
valueWBUBKdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights*
dtype0*
_output_shapes
:
y
!save/RestoreV2_7/shape_and_slicesConst*$
valueBB32 32 0,32:0,32*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes

:  *
dtypes
2
╢
save/Assign_7AssignRdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0save/RestoreV2_7*
T0*e
_class[
YWloc:@dnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0*
validate_shape(*
_output_shapes

:  *
use_locking(
u
save/RestoreV2_8/tensor_namesConst*$
valueBBdnn/logits/bias*
dtype0*
_output_shapes
:
o
!save/RestoreV2_8/shape_and_slicesConst*
valueBB1 0,1*
dtype0*
_output_shapes
:
Ш
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
_output_shapes
:*
dtypes
2
║
save/Assign_8Assigndnn/logits/bias/part_0save/RestoreV2_8*
T0*)
_class
loc:@dnn/logits/bias/part_0*
validate_shape(*
_output_shapes
:*
use_locking(
w
save/RestoreV2_9/tensor_namesConst*&
valueBBdnn/logits/kernel*
dtype0*
_output_shapes
:
w
!save/RestoreV2_9/shape_and_slicesConst*"
valueBB50 1 0,50:0,1*
dtype0*
_output_shapes
:
Ь
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes

:2
┬
save/Assign_9Assigndnn/logits/kernel/part_0save/RestoreV2_9*
use_locking(*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
validate_shape(*
_output_shapes

:2
r
save/RestoreV2_10/tensor_namesConst* 
valueBBglobal_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Щ
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2	*
_output_shapes
:
в
save/Assign_10Assignglobal_stepsave/RestoreV2_10*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
╔
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10
-
save/restore_allNoOp^save/restore_shardЎ
є
*
_make_dataset_ad3a09d5
BatchDataset{
&TensorSliceDataset/tensors/component_0Const*=
value4B2
"(l  S  О  K  W  к  ;  ▌  p    *
dtype0{
&TensorSliceDataset/tensors/component_1Const*=
value4B2
"(Й  =    2  └   $  C  H  █  э  *
dtype0╧
&TensorSliceDataset/tensors/component_2Const*Р
valueЖBГ
BSoutheasternBSoutheasternBBig TenBBig 12BAtlantic CoastBAtlantic CoastBPac-12BMountain WestBBig TenBSoutheastern*
dtype0{
&TensorSliceDataset/tensors/component_3Const*=
value4B2
"(▌  ╘  ╤  ╤  ╤  ╧  ╧  ┘  ┘  ╫  *
dtype0{
&TensorSliceDataset/tensors/component_4Const*=
value4B2
"(/   +   -         -      -   .   $   *
dtype0{
&TensorSliceDataset/tensors/component_5Const*=
value4B2
"(                                        *
dtype0{
&TensorSliceDataset/tensors/component_6Const*=
value4B2
"('   #   -                   .      *
dtype0{
&TensorSliceDataset/tensors/component_7Const*=
value4B2
"(э          m            о   ╔      *
dtype0{
&TensorSliceDataset/tensors/component_8Const*=
value4B2
"(                              *
dtype0{
&TensorSliceDataset/tensors/component_9Const*=
value4B2
"(  А   №   д   ы   ╤  л   Д   с   Л   *
dtype0|
'TensorSliceDataset/tensors/component_10Const*=
value4B2
"(                              *
dtype0|
'TensorSliceDataset/tensors/component_11Const*=
value4B2
"(  y   Д  Ф      o  Ш   \  O   *
dtype0Д
'TensorSliceDataset/tensors/component_12Const*E
value<B:
BSFOBSDGBSDGBWASBATLBPHIBCINBDENBINDBOAK*
dtype0|
'TensorSliceDataset/tensors/component_13Const*=
value4B2
"(4   Q   Z         M   -   /   C   4   *
dtype0|
'TensorSliceDataset/tensors/component_14Const*=
value4B2
"(ё   З'  .  D  у  ┼     ╔  Ы+  с  *
dtype0|
'TensorSliceDataset/tensors/component_15Const*=
value4B2
"(                                    *
dtype0Ъ
TensorSliceDatasetTensorSliceDataset/TensorSliceDataset/tensors/component_0:output:0/TensorSliceDataset/tensors/component_1:output:0/TensorSliceDataset/tensors/component_2:output:0/TensorSliceDataset/tensors/component_3:output:0/TensorSliceDataset/tensors/component_4:output:0/TensorSliceDataset/tensors/component_5:output:0/TensorSliceDataset/tensors/component_6:output:0/TensorSliceDataset/tensors/component_7:output:0/TensorSliceDataset/tensors/component_8:output:0/TensorSliceDataset/tensors/component_9:output:00TensorSliceDataset/tensors/component_10:output:00TensorSliceDataset/tensors/component_11:output:00TensorSliceDataset/tensors/component_12:output:00TensorSliceDataset/tensors/component_13:output:00TensorSliceDataset/tensors/component_14:output:00TensorSliceDataset/tensors/component_15:output:0*3
output_shapes"
 : : : : : : : : : : : : : : : : *%
Toutput_types
2A
BatchDataset/batch_sizeConst*
value	B	 R
*
dtype0	Й
BatchDatasetBatchDatasetTensorSliceDataset:handle:0 BatchDataset/batch_size:output:0*$
output_types
2*Е
output_shapesє
Ё:         :         :         :         :         :         :         :         :         :         :         :         :         :         :         :         "%
BatchDatasetBatchDataset:handle:0"" 
global_step

global_step:0"&

summary_op

Merge/MergeSummary:0"я
model_variables█╪
°
Zdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights  "2wdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal:0
┌
Tdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0:0Ydnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/AssignYdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/read:0"Y
Kdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights    "  2qdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal:0"╫
	summaries╔
╞
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"╡
trainable_variablesЭЪ
°
Zdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights  "2wdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal:0
┌
Tdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0:0Ydnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/AssignYdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/read:0"Y
Kdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights    "  2qdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal:0
┘
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel92  "922<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
├
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias2 "221dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
┘
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kernel2d  "2d2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
├
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/biasd "d21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
┘
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kerneld2  "d22<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
├
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias2 "221dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
╢
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel2  "225dnn/logits/kernel/part_0/Initializer/random_uniform:0
а
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"
init_op

group_deps_1"U
ready_for_local_init_op:
8
6report_uninitialized_variables_1/boolean_mask/Gather:0"
	eval_step

eval_step:0"└∙
cond_contextо∙к∙
Т
>dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0 *╠
Idnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_t:0
╨	
@dnn/head/assert_range/assert_less/Assert/AssertGuard/cond_text_1>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0*К
dnn/head/ToFloat:0
dnn/head/assert_range/Const:0
'dnn/head/assert_range/assert_less/All:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch:0
Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1:0
Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_0:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_1:0
Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/data_3:0
Kdnn/head/assert_range/assert_less/Assert/AssertGuard/control_dependency_1:0
>dnn/head/assert_range/assert_less/Assert/AssertGuard/pred_id:0
?dnn/head/assert_range/assert_less/Assert/AssertGuard/switch_f:0g
dnn/head/assert_range/Const:0Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_2:0o
'dnn/head/assert_range/assert_less/All:0Ddnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch:0\
dnn/head/ToFloat:0Fdnn/head/assert_range/assert_less/Assert/AssertGuard/Assert/Switch_1:0
о
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/cond_textXdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0 *Ъ
cdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency:0
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_t:0
З
Zdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/cond_text_1Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0*є
dnn/head/ToFloat:0
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
`dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:0
ednn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Xdnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/pred_id:0
Ydnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/switch_f:0г
Adnn/head/assert_range/assert_non_negative/assert_less_equal/All:0^dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch:0v
dnn/head/ToFloat:0`dnn/head/assert_range/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
╠
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textwdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *█
Щdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Щdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Щdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Пdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Лdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Цdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Сdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Нdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Мdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Зdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Сdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Рdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Рdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Кdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Вdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Аdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0я
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0Цdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1ь
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
╤	
ydnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*р
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
|dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0°
|dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
й
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *Ж
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0▓
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
╨
_dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*н
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
udnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
vdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Аdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Гdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
Еdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
|dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
wdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
xdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
]dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
^dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0▌
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0Еdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0┌
Rdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/rank:0Гdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0э
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0Фdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0ъ
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0Тdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
║
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_textZdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *а
ednn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0
ф
\dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*╩
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
gdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
Zdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0║
Tdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0╣
Sdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/values/shape:0bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0╢
Pdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0bdnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0┐
[dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0`dnn/head/metrics/label/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
т 
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *т
Юdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Юdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Юdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Фdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Рdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Ыdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Цdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Тdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Сdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Мdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Цdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Хdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Хdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Пdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Зdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Еdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0Ў
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1∙
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0Ыdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Е

~dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Е
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
Бdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0Г
Бdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
█
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_textbdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *й
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0╝
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
ь
ddnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*║
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
zdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
{dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Еdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Иdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
Кdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
Бdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
|dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
}dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
bdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
cdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0Ї
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0Чdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ч
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/rank:0Кdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0ф
Wdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/rank:0Иdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0ў
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0Щdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
╪
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *п
jdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0
Ё
adnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*╟
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
ldnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
_dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0└
Udnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_scalar:0gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0├
Xdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/values/shape:0gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0╔
`dnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0ednn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0─
Ydnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/weights/shape:0gdnn/head/metrics/prediction/mean/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
┬
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/cond_textFdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t:0 *ф
Qdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency:0
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0
Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_t:0
Ў

Hdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/cond_text_1Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f:0*Ш	
dnn/head/metrics/auc/Cast_1/x:0
/dnn/head/metrics/auc/assert_greater_equal/All:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0
Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0
Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_0:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_1:0
Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/data_3:0
Sdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/control_dependency_1:0
Fdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/pred_id:0
Gdnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0
/dnn/head/metrics/auc/assert_greater_equal/All:0Ldnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0q
dnn/head/metrics/auc/Cast_1/x:0Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0q
dnn/head/predictions/logistic:0Ndnn/head/metrics/auc/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0
░
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/cond_textCdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t:0 *█
Ndnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency:0
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0
Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_t:0
├

Ednn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/cond_text_1Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f:0*ю
dnn/head/metrics/auc/Cast_2/x:0
,dnn/head/metrics/auc/assert_less_equal/All:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/data_3:0
Pdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Cdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/pred_id:0
Ddnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0n
dnn/head/predictions/logistic:0Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0n
dnn/head/metrics/auc/Cast_2/x:0Kdnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0y
,dnn/head/metrics/auc/assert_less_equal/All:0Idnn/head/metrics/auc/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
°
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textpdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *Ь
Тdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
Тdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
Тdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Иdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Дdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Пdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Кdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Жdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Еdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Аdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Кdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Йdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Йdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Гdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
{dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0с
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0Пdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1▐
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Л	
rdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*п
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0ъ
udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
у
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_textVdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *╒
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0д
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
Є
Xdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*ф
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
ndnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
odnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
pdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
qdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Vdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
Wdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank:0
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0▄
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0Лdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0╬
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/rank:0~dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0╦
Kdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/rank:0|dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0▀
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0Нdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Р
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_textSdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *Л
^dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0
а
Udnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*Ы
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
`dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
Sdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0▒
Tdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0Ydnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0м
Mdnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/weights/shape:0[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0и
Idnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/is_scalar:0[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0л
Ldnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/values/shape:0[dnn/head/metrics/auc/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
и
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/cond_textWdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t:0 *Ч
bdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency:0
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0
Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_t:0
╝
Ydnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/cond_text_1Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f:0*л
0dnn/head/metrics/auc_precision_recall/Cast_1/x:0
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0
_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0
_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_0:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_1:0
]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/data_3:0
ddnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/control_dependency_1:0
Wdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/pred_id:0
Xdnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0В
dnn/head/predictions/logistic:0_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_1:0б
@dnn/head/metrics/auc_precision_recall/assert_greater_equal/All:0]dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch:0У
0dnn/head/metrics/auc_precision_recall/Cast_1/x:0_dnn/head/metrics/auc_precision_recall/assert_greater_equal/Assert/AssertGuard/Assert/Switch_2:0
Ц
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/cond_textTdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t:0 *О
_dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency:0
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0
Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_t:0
И
Vdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/cond_text_1Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f:0*А
0dnn/head/metrics/auc_precision_recall/Cast_2/x:0
=dnn/head/metrics/auc_precision_recall/assert_less_equal/All:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0
\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_0:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_1:0
Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/data_3:0
adnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/control_dependency_1:0
Tdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/pred_id:0
Udnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/switch_f:0
dnn/head/predictions/logistic:0Р
0dnn/head/metrics/auc_precision_recall/Cast_2/x:0\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_2:0
dnn/head/predictions/logistic:0\dnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch_1:0Ы
=dnn/head/metrics/auc_precision_recall/assert_less_equal/All:0Zdnn/head/metrics/auc_precision_recall/assert_less_equal/Assert/AssertGuard/Assert/Switch:0
¤!
Бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textБdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *ы
гdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
гdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
гdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
Щdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
Хdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
Ыdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
Чdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
Цdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
Сdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
Ыdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
Ъdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
Ъdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
Фdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
Мdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
Кdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0Г
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1А
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
┐

Гdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1Бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*н
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
Жdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0О
Жdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Н
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_textgdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0 *╠
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_t:0╞
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Switch_1:1
Н
idnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/cond_text_1gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0*╠
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
Аdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
Кdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Нdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
Пdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
Жdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Бdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
Вdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
gdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/pred_id:0
hdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/switch_f:0
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank:0
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0Б
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0Юdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0ю
\dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/rank:0Нdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0■
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0Ьdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ё
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/rank:0Пdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
Ў
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_textddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0 *╛
odnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency:0
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_t:0
№
fdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/cond_text_1ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0*─
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0
ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0
ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0
ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_0:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_1:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_2:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_4:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_5:0
jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/data_7:0
qdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/control_dependency_1:0
ddnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/pred_id:0
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/switch_f:0
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0═
]dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/values/shape:0ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_2:0╩
Zdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_scalar:0ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_3:0╙
ednn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/is_valid_shape/Merge:0jdnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch:0╬
^dnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/weights/shape:0ldnn/head/metrics/auc_precision_recall/broadcast_weights_1/assert_broadcastable/AssertGuard/Assert/Switch_1:0"═
local_variables╣╢
░
#dnn/head/metrics/label/mean/total:0(dnn/head/metrics/label/mean/total/Assign(dnn/head/metrics/label/mean/total/read:025dnn/head/metrics/label/mean/total/Initializer/zeros:0
░
#dnn/head/metrics/label/mean/count:0(dnn/head/metrics/label/mean/count/Assign(dnn/head/metrics/label/mean/count/read:025dnn/head/metrics/label/mean/count/Initializer/zeros:0
╕
%dnn/head/metrics/average_loss/total:0*dnn/head/metrics/average_loss/total/Assign*dnn/head/metrics/average_loss/total/read:027dnn/head/metrics/average_loss/total/Initializer/zeros:0
╕
%dnn/head/metrics/average_loss/count:0*dnn/head/metrics/average_loss/count/Assign*dnn/head/metrics/average_loss/count/read:027dnn/head/metrics/average_loss/count/Initializer/zeros:0
и
!dnn/head/metrics/accuracy/total:0&dnn/head/metrics/accuracy/total/Assign&dnn/head/metrics/accuracy/total/read:023dnn/head/metrics/accuracy/total/Initializer/zeros:0
и
!dnn/head/metrics/accuracy/count:0&dnn/head/metrics/accuracy/count/Assign&dnn/head/metrics/accuracy/count/read:023dnn/head/metrics/accuracy/count/Initializer/zeros:0
─
(dnn/head/metrics/prediction/mean/total:0-dnn/head/metrics/prediction/mean/total/Assign-dnn/head/metrics/prediction/mean/total/read:02:dnn/head/metrics/prediction/mean/total/Initializer/zeros:0
─
(dnn/head/metrics/prediction/mean/count:0-dnn/head/metrics/prediction/mean/count/Assign-dnn/head/metrics/prediction/mean/count/read:02:dnn/head/metrics/prediction/mean/count/Initializer/zeros:0
╕
%dnn/head/metrics/auc/true_positives:0*dnn/head/metrics/auc/true_positives/Assign*dnn/head/metrics/auc/true_positives/read:027dnn/head/metrics/auc/true_positives/Initializer/zeros:0
╝
&dnn/head/metrics/auc/false_negatives:0+dnn/head/metrics/auc/false_negatives/Assign+dnn/head/metrics/auc/false_negatives/read:028dnn/head/metrics/auc/false_negatives/Initializer/zeros:0
╕
%dnn/head/metrics/auc/true_negatives:0*dnn/head/metrics/auc/true_negatives/Assign*dnn/head/metrics/auc/true_negatives/read:027dnn/head/metrics/auc/true_negatives/Initializer/zeros:0
╝
&dnn/head/metrics/auc/false_positives:0+dnn/head/metrics/auc/false_positives/Assign+dnn/head/metrics/auc/false_positives/read:028dnn/head/metrics/auc/false_positives/Initializer/zeros:0
№
6dnn/head/metrics/auc_precision_recall/true_positives:0;dnn/head/metrics/auc_precision_recall/true_positives/Assign;dnn/head/metrics/auc_precision_recall/true_positives/read:02Hdnn/head/metrics/auc_precision_recall/true_positives/Initializer/zeros:0
А
7dnn/head/metrics/auc_precision_recall/false_negatives:0<dnn/head/metrics/auc_precision_recall/false_negatives/Assign<dnn/head/metrics/auc_precision_recall/false_negatives/read:02Idnn/head/metrics/auc_precision_recall/false_negatives/Initializer/zeros:0
№
6dnn/head/metrics/auc_precision_recall/true_negatives:0;dnn/head/metrics/auc_precision_recall/true_negatives/Assign;dnn/head/metrics/auc_precision_recall/true_negatives/read:02Hdnn/head/metrics/auc_precision_recall/true_negatives/Initializer/zeros:0
А
7dnn/head/metrics/auc_precision_recall/false_positives:0<dnn/head/metrics/auc_precision_recall/false_positives/Assign<dnn/head/metrics/auc_precision_recall/false_positives/read:02Idnn/head/metrics/auc_precision_recall/false_positives/Initializer/zeros:0
T
mean/total:0mean/total/Assignmean/total/read:02mean/total/Initializer/zeros:0
T
mean/count:0mean/count/Assignmean/count/read:02mean/count/Initializer/zeros:0
P
eval_step:0eval_step/Assigneval_step/read:02eval_step/Initializer/zeros:0"!
local_init_op

group_deps_2"Е
	variablesўЇ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
°
Zdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights  "2wdnn/input_from_feature_columns/input_layer/Conference_embedding/embedding_weights/part_0/Initializer/truncated_normal:0
┌
Tdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0:0Ydnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/AssignYdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/read:0"Y
Kdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights    "  2qdnn/input_from_feature_columns/input_layer/Team_embedding/embedding_weights/part_0/Initializer/truncated_normal:0
┘
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign&dnn/hiddenlayer_0/kernel/part_0/read:0"&
dnn/hiddenlayer_0/kernel92  "922<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:0
├
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign$dnn/hiddenlayer_0/bias/part_0/read:0"!
dnn/hiddenlayer_0/bias2 "221dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:0
┘
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign&dnn/hiddenlayer_1/kernel/part_0/read:0"&
dnn/hiddenlayer_1/kernel2d  "2d2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:0
├
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign$dnn/hiddenlayer_1/bias/part_0/read:0"!
dnn/hiddenlayer_1/biasd "d21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:0
┘
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign&dnn/hiddenlayer_2/kernel/part_0/read:0"&
dnn/hiddenlayer_2/kerneld2  "d22<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:0
├
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign$dnn/hiddenlayer_2/bias/part_0/read:0"!
dnn/hiddenlayer_2/bias2 "221dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:0
╢
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assigndnn/logits/kernel/part_0/read:0"
dnn/logits/kernel2  "225dnn/logits/kernel/part_0/Initializer/random_uniform:0
а
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assigndnn/logits/bias/part_0/read:0"
dnn/logits/bias "2*dnn/logits/bias/part_0/Initializer/zeros:0"
ready_op


concat:0" 
asset_filepathsы
ш
xdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init/asset_filepath:0
ldnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init/asset_filepath:0"▀
table_initializer╔
╞
gdnn/input_from_feature_columns/input_layer/Conference_embedding/Conference_lookup/hash_table/table_init
[dnn/input_from_feature_columns/input_layer/Team_embedding/Team_lookup/hash_table/table_init"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"*
losses 

dnn/head/weighted_loss/Sum:0шХ╦^░       ЦM	4	у№#HОК╓AРN*б

loss╜ДA

accuracy_baselineЪЩ?


aucйк:?

prediction/mean│_ж>


label/mean═╠╠>

average_loss╚J╙?

auc_precision_recallкк"?

accuracy333?Ъ#PШ