(* ::Package:: *)

(*
   Implementation of ideas from Real NVP
   Ref: https://arxiv.org/pdf/1605.08803.pdf
   Dinh, Sohl-Dickstein, Bengio, 2017.
   This Mathematica implementation is not identical with reference implementation, but
   does use ideas from their paper.
   
   Achieves reasonable generative model on MNIST, good clear images, digit-like but not always
   recognisable as specific digit.
   When trained on CelebA dataset around 20% of samples look very realistic, about 50% strongly resemble
   faces (but notable generative artefacts) and 20% look quite distorted. Model has log likelihood of around -2000.
   Did experiment with more coupling layers, more powerful coupling nets, increasing power of coupling nets through
   each layer, and also implementing multi-scale (ie factoring out at each level). These changes seem to make
   little difference either in isolation or in combination.
*)


gauss = NetChain[{
   ElementwiseLayer[-0.5*#^2-Log[Sqrt[2*Pi]]&],\
   SummationLayer[]}];


MaskLayer[ mask_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


CheckerboardMask[ {channels_, row_, col_ } ] :=
   ReplacePart[ ConstantArray[ 0, {channels, row, col} ], {ch_, r_, c_}?(Function[v,If[OddQ[v[[2]]],OddQ[v[[3]]],EvenQ[v[[3]]]]])->1 ]


StableScaleNet = NetGraph[{"nonlinearity"->Tanh,"c1"->ConstantArrayLayer[],"c2"->ConstantArrayLayer[],"times"->ThreadingLayer[Times],"plus"->ThreadingLayer[Plus]},{{"nonlinearity","c1"}->"times",{"times","c2"}->"plus"}];


CouplingLayer[ passThrough_ ] := NetGraph[{
   "passDirect"->MaskLayer[passThrough],
   "change"->MaskLayer[1-passThrough],
   "mu"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[Length[passThrough],{3,3},"PaddingSize"->1],StableScaleNet},
   "logscale"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[Length[passThrough],{3,3},"PaddingSize"->1],StableScaleNet},
   "scale"->ElementwiseLayer[Exp],
   "invmu"->ElementwiseLayer[-#&],
   "invscale"->ElementwiseLayer[1/#&],
   "inv1"->ThreadingLayer[Plus],
   "inv2"->ThreadingLayer[Times],
   "changed"->MaskLayer[1-passThrough],
   "new"->ThreadingLayer[Plus],
   "jacobian1"->ElementwiseLayer[-#&],
   "jacobian2"->MaskLayer[1-passThrough],
   "jacobian3"->SummationLayer[]
},{
   "passDirect"->{"mu","logscale"},
   "logscale"->"scale"->"invscale",
   "mu"->"invmu",
   {"change","invmu"}->"inv1",
   {"inv1","invscale"}->"inv2"->"changed",
   {"changed","passDirect"}->"new"->NetPort["Transformed"],
   "logscale"->"jacobian1"->"jacobian2"->"jacobian3"->NetPort["Jacobian"]
}]


CouplingLayerLRM = {
   {"passDirect","mask"}->0,
   {"change","mask"}->0,
   {"changed","mask"}->0,
   {"jacobian2","mask"}->0 };


ChannelMask[ { channels_, row_, col_ } ] :=
   ReplacePart[ ConstantArray[ 0,{channels,row,col} ], {channel_/;channel<=channels/2,_,_}->1]


SqueezeLayer2x2[ inChannels_ ] := NetGraph[{
   "s1"->ConvolutionLayer[inChannels,{2,2},"Stride"->2,"Biases"->ConstantArray[0,{inChannels}],"Weights"->Table[
      If[o==i,{{1,0},{0,0}},{{0,0},{0,0}}],{o,inChannels},{i,inChannels}]],
   "s2"->ConvolutionLayer[inChannels,{2,2},"Stride"->2,"Biases"->ConstantArray[0,{inChannels}],"Weights"->Table[
      If[o==i,{{0,1},{0,0}},{{0,0},{0,0}}],{o,inChannels},{i,inChannels}]],
   "s3"->ConvolutionLayer[inChannels,{2,2},"Stride"->2,"Biases"->ConstantArray[0,{inChannels}],"Weights"->Table[
      If[o==i,{{0,0},{1,0}},{{0,0},{0,0}}],{o,inChannels},{i,inChannels}]],
   "s4"->ConvolutionLayer[inChannels,{2,2},"Stride"->2,"Biases"->ConstantArray[0,{inChannels}],"Weights"->Table[
      If[o==i,{{0,0},{0,1}},{{0,0},{0,0}}],{o,inChannels},{i,inChannels}]],
   "cat"->CatenateLayer[]
},{
{"s1","s2","s3","s4"}->"cat"
}];


SqueezeLayerLRM = {
   {"s1"}->0,
   {"s2"}->0,
   {"s3"}->0,
   {"s4"}->0 };


RealNVPBlock[ checkerboard_, channelmask_, inSqueezeChannels_ ] := NetGraph[{
   "c1"->CouplingLayer[ checkerboard ],
   "c2"->CouplingLayer[ 1 - checkerboard ],
   "sq1"->SqueezeLayer2x2[ inSqueezeChannels ],
   "c3"->CouplingLayer[ channelmask ],
   "c4"->CouplingLayer[ 1 - channelmask ],
   "jacobian"->ThreadingLayer[Plus]},{
   NetPort[{"c1","Transformed"}]->"c2",
   NetPort[{"c2","Transformed"}]->"sq1",
   "sq1"->"c3",
   NetPort[{"c3","Transformed"}]->"c4",
   NetPort[{"c4","Transformed"}]->NetPort["Transformed"],
   {NetPort["c1","Jacobian"],NetPort["c2","Jacobian"],NetPort["c3","Jacobian"],NetPort["c4","Jacobian"]}->
      "jacobian"->NetPort["Jacobian"]
   }];


PrependLRM[ lrms_, layerName_ ] :=
   Map[ Prepend[ #[[1]], layerName ]->0 &, lrms ]


RealNVPBlockLRM = Join[
   PrependLRM[ CouplingLayerLRM, "c1" ],
   PrependLRM[ CouplingLayerLRM, "c2" ],
   PrependLRM[ SqueezeLayerLRM, "sq1" ],
   PrependLRM[ CouplingLayerLRM, "c3" ],
   PrependLRM[ CouplingLayerLRM, "c4" ]
];


(* Note you do need to ensure the block number is appropriate to the dimensions of the Input *)
RealNVP[blocks_,channelCoupling_,dims_] := Module[{els = Apply[Times,dims]},NetGraph[Join[
   Table["block"<>ToString[ k+1 ]->
      RealNVPBlock[ 
         CheckerboardMask[{dims[[1]]*4^k,dims[[2]]/2^k,dims[[3]]/2^k}], 
         ChannelMask[{dims[[1]]*4^(k+1),dims[[2]]/2^(k+1),dims[[3]]/2^(k+1)}], 4^k],{k,0,blocks-1}],
   {"reshape"->ReshapeLayer[{els,1,1}]},
   Table["c"<>ToString[k]->CouplingLayer[ ChannelMask[{els,1,1}] ],{k,channelCoupling}],
   Table["d"<>ToString[k]->CouplingLayer[ 1 - ChannelMask[{els,1,1}] ],{k,channelCoupling}],{
   "g"->gauss,
   "th"->ThreadingLayer[Plus],
   "loss"->ElementwiseLayer[ -#& ]}]
,Join[
   Table[NetPort[{"block"<>ToString[k],"Transformed"}]->"block"<>ToString[k+1],{k,1,blocks-1}],
{   If[blocks>0,NetPort[{"block"<>ToString[blocks],"Transformed"}]->"reshape",Nothing],
   "reshape"->If[channelCoupling>0,"c1","g"],
   Table[NetPort[{"c"<>ToString[k],"Transformed"}]->"d"<>ToString[k],{k,channelCoupling}],
   Table[NetPort[{"d"<>ToString[k],"Transformed"}]->"c"<>ToString[k+1],{k,channelCoupling-1}],
   If[channelCoupling>0,NetPort[{"d"<>ToString[channelCoupling],"Transformed"}]->"g",Nothing],
   Join[Table[NetPort[{"block"<>ToString[k],"Jacobian"}],{k,blocks}],
      {"g"},Table[NetPort[{"c"<>ToString[k],"Jacobian"}],{k,channelCoupling}],Table[NetPort[{"d"<>ToString[k],"Jacobian"}],{k,channelCoupling}]]->"th"->"loss"->NetPort["Loss"]
   }],
   "Input"->dims
]];


RealNVPLRM[ blocks_, channelCoupling_ ] := Flatten@Join[
   Table[PrependLRM[ RealNVPBlockLRM, "block"<>ToString[k] ],{k,blocks}],
   Table[PrependLRM[ CouplingLayerLRM, "c"<>ToString[k] ],{k,channelCoupling}],
   Table[PrependLRM[ CouplingLayerLRM, "d"<>ToString[k] ],{k,channelCoupling}]
];


images = ResourceData["MNIST","TrainingData"][[All,1]];


resize = ImageResize[#,{32,32}]&/@images;


data = {ImageData[#]}&/@resize;data//Dimensions;


data1 = data+Table[RandomReal[]/10,{60000},{1},{32},{32}];


train = NetTrain[ RealNVP, 
   Association["Input"->#]&/@data1,
   LearningRateMultipliers->RealNVPLRM[3,1],
   LossFunction->"Loss"
];


ReverseCouplingLayer[net_,z_,mask_]:=(
   mu=NetTake[net,"mu"][z];
   scale=NetTake[net,"scale"][z];
   ch=z*scale+mu;
   tot=ch*(1-mask)+z*mask)


ReverseRealNVP[ z_ ] := (
   oc2i = ReverseCouplingLayer[ NetExtract[ train, {"c2"} ], z, 1-ChannelMask[{1024,1,1}] ];
   oc1i = ReverseCouplingLayer[ NetExtract[ train, {"c1"} ], oc2i, ChannelMask[{1024,1,1}] ];
   reshape = ReshapeLayer[{64,4,4}][oc1i];
   revblock3 = ReverseRealNVPBlock[ "block3", { 16, 8, 8 }, reshape ];
   revblock2 = ReverseRealNVPBlock[ "block2", { 4, 16, 16 }, revblock3 ];
   revblock1 = ReverseRealNVPBlock[ "block1", { 1, 32, 32 }, revblock2 ]
)


(* Note dims is dims of the input to the squeeze operation *)
ReverseSqueezeLayer[ inp_, dims_ ] := Module[{tmp = ConstantArray[0,dims]},
   tmp[[All,1;;dims[[2]];;2,1;;dims[[3]];;2]] = inp[[;;dims[[1]]]];
   tmp[[All,1;;dims[[2]];;2,2;;dims[[3]];;2]] = inp[[dims[[1]]+1;;dims[[1]]*2]];
   tmp[[All,2;;dims[[2]];;2,1;;dims[[3]];;2]] = inp[[dims[[1]]*2+1;;dims[[1]]*3]];
   tmp[[All,2;;dims[[2]];;2,2;;dims[[3]];;2]] = inp[[dims[[1]]*3+1;;]];
   tmp
];


(*
   Note dims is the dimensions of the input to the block
*)
ReverseRealNVPBlock[ blockName_, dims_, z_ ] := (
   c4i = ReverseCouplingLayer[ NetExtract[ train, {blockName,"c4"} ], z, 1 - ChannelMask[{dims[[1]]*4,dims[[2]]/2,dims[[3]]/2}] ];
   c3i = ReverseCouplingLayer[ NetExtract[ train, {blockName,"c3"} ], c4i, ChannelMask[{dims[[1]]*4,dims[[2]]/2,dims[[3]]/2}] ];
   desq = ReverseSqueezeLayer[ c3i, dims ];
   c2i = ReverseCouplingLayer[ NetExtract[ train, {blockName,"c2"} ], desq, 1 - CheckerboardMask[dims] ];
   c1i = ReverseCouplingLayer[ NetExtract[ train, {blockName,"c1"} ], c2i, CheckerboardMask[dims] ]
)


gaussiansample[] := Table[RandomVariate@NormalDistribution[0,1],{1024},{1},{1}]


sample[] := ReverseRealNVP[ gaussiansample[] ];
