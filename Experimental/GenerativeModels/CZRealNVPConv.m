(* ::Package:: *)

gauss = NetChain[{
   ElementwiseLayer[-0.5*#^2-Log[Sqrt[2*Pi]]&],\
   SummationLayer[]}];


MaskLayer[ mask_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


CheckerboardMask[ {channels_, row_, col_ } ] :=
   ReplacePart[ ConstantArray[ 0, {channels, row, col} ], {ch_, r_, c_}?(Function[v,If[OddQ[v[[2]]],OddQ[v[[3]]],EvenQ[v[[3]]]]])->1 ]


StableScaleNet = NetGraph[{"nonlinearity"->Tanh,"c1"->ConstantArrayLayer[{1}],"c2"->ConstantArrayLayer[{1}],"times"->ThreadingLayer[Times],"plus"->ThreadingLayer[Plus]},{{"nonlinearity","c1"}->"times",{"times","c2"}->"plus"}];


CouplingLayer[ passThrough_ ] := NetGraph[{
   "passDirect"->MaskLayer[passThrough],
   "change"->MaskLayer[1-passThrough],
   "mu"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[Length[passThrough],{3,3},"PaddingSize"->1]},
   "logscale"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[Length[passThrough],{3,3},"PaddingSize"->1](*,StableScaleNet*)},
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


RealNVP = NetGraph[{
   "block1"->RealNVPBlock[ CheckerboardMask[{1,32,32}], ChannelMask[{4,16,16}], 1],
   "block2"->RealNVPBlock[ CheckerboardMask[{4,16,16}], ChannelMask[{16,8,8}], 4],
   "block3"->RealNVPBlock[ CheckerboardMask[{16,8,8}], ChannelMask[{64,4,4}], 16],
   "reshape"->ReshapeLayer[{1024,1,1}],
   "c1"->CouplingLayer[ ChannelMask[{1024,1,1}] ],
   "c2"->CouplingLayer[ 1 - ChannelMask[{1024,1,1}] ],
   "g"->gauss,
   "th"->ThreadingLayer[Plus],
   "loss"->ElementwiseLayer[ -#& ]
},{
   NetPort[{"block1","Transformed"}]->"block2",
   NetPort[{"block2","Transformed"}]->"block3",
   NetPort[{"block3","Transformed"}]->"reshape"->"c1",
   NetPort[{"c1","Transformed"}]->"c2",
   NetPort[{"c2","Transformed"}]->"g",
   {NetPort[{"block1","Jacobian"}],NetPort[{"block2","Jacobian"}],NetPort[{"block3","Jacobian"}],"g",NetPort[{"c1","Jacobian"}],NetPort[{"c2","Jacobian"}]}->"th"->"loss"->NetPort["Loss"]
},"Input"->{1,32,32}];


RealNVPLRM = Join[
   PrependLRM[ RealNVPBlockLRM, "block1" ],
   PrependLRM[ RealNVPBlockLRM, "block2" ],
   PrependLRM[ RealNVPBlockLRM, "block3" ],
   PrependLRM[ CouplingLayerLRM, "c1" ],
   PrependLRM[ CouplingLayerLRM, "c2" ]
];


images = ResourceData["MNIST","TrainingData"][[5924;;12665,1]];


resize = ImageResize[#,{32,32}]&/@images;


data = {ImageData[#]}&/@resize;data//Dimensions;


data1 = data+Table[RandomReal[]/10,{6742},{1},{32},{32}];


train = NetTrain[ RealNVP, 
   Association["Input"->#]&/@data1,
   LearningRateMultipliers->RealNVPLRM,
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
