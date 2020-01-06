(* ::Package:: *)

gauss = NetChain[{
   ElementwiseLayer[-0.5*#^2-Log[Sqrt[2*Pi]]&],\
   SummationLayer[]}];


MaskLayer[ mask_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


checkerboard1x8x8 = ConstantArray[0,{1,8,8}];
checkerboard1x8x8[[1,1;;8;;2,1;;8;;2]] = 1;
checkerboard1x8x8[[1,2;;8;;2,2;;8;;2]] = 1;


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


channelmask4x4x4 = ConstantArray[0,{4,4,4}];
channelmask4x4x4[[1;;2,All,All]] = 1;


channelmask64x1x1 = ConstantArray[0,{64,1,1}];
channelmask64x1x1[[1;;32,1,1]] = 1;


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


RealNVPBlock = NetGraph[{
   "c1"->CouplingLayer[ checkerboard1x8x8 ],
   "c2"->CouplingLayer[ 1 - checkerboard1x8x8 ],
   "s1"->mysqueeze[ 1 ],
   "c3"->CouplingLayer[ channelmask4x4x4 ],
   "c4"->CouplingLayer[ 1 - channelmask4x4x4 ],
   "jacobian"->ThreadingLayer[Plus]},{
   NetPort[{"c1","Transformed"}]->"c2",
   NetPort[{"c2","Transformed"}]->"s1",
   "s1"->"c3",
   NetPort[{"c3","Transformed"}]->"c4",
   NetPort[{"c4","Transformed"}]->NetPort["Transformed"],
   {NetPort["c1","Jacobian"],NetPort["c2","Jacobian"],NetPort["c3","Jacobian"],NetPort["c4","Jacobian"]}->
      "jacobian"->NetPort["Jacobian"]
   }];


RealNVP = NetGraph[{
   "block"->RealNVPBlock,   
   "reshape"->ReshapeLayer[{64,1,1}],
   "c1"->CouplingLayer[ channelmask64x1x1 ],
   "c2"->CouplingLayer[ 1 - channelmask64x1x1 ],
   "g"->gauss,
   "th"->ThreadingLayer[Plus],
   "loss"->ElementwiseLayer[ -#& ]
},{
   NetPort[{"block","Transformed"}]->"reshape"->"c1",
   NetPort[{"c1","Transformed"}]->"c2",
   NetPort[{"c2","Transformed"}]->"g",
   {NetPort[{"block","Jacobian"}],"g",NetPort[{"c1","Jacobian"}],NetPort[{"c2","Jacobian"}]}->"th"->"loss"->NetPort["Loss"]
},"Input"->{1,8,8}];


images = ResourceData["MNIST","TrainingData"][[5924;;12665,1]];


resize = ImageResize[#,{8,8}]&/@images;


data = {ImageData[#]}&/@resize;data//Dimensions;


data1 = data+Table[RandomReal[]/10,{6742},{1},{8},{8}];


train = NetTrain[ RealNVP, Association["Input"->#]&/@data1,LearningRateMultipliers->{
   {"block","s1","s1"}->0,
   {"block","s1","s2"}->0,
   {"block","s1","s3"}->0,
   {"block","s1","s4"}->0,


   {"block","c1","passDirect","mask"}->0,
   {"block","c1","change","mask"}->0,
   {"block","c1","changed","mask"}->0,
   {"block","c1","jacobian2","mask"}->0,

   {"block","c2","passDirect","mask"}->0,
   {"block","c2","change","mask"}->0,
   {"block","c2","changed","mask"}->0,
   {"block","c2","jacobian2","mask"}->0,
   
   {"block","c3","passDirect","mask"}->0,
   {"block","c3","change","mask"}->0,
   {"block","c3","changed","mask"}->0,
   {"block","c3","jacobian2","mask"}->0,
   
   {"block","c4","passDirect","mask"}->0,
   {"block","c4","change","mask"}->0,
   {"block","c4","changed","mask"}->0,
   {"block","c4","jacobian2","mask"}->0,

   {"c1","passDirect","mask"}->0,
   {"c1","change","mask"}->0,
   {"c1","changed","mask"}->0,
   {"c1","jacobian2","mask"}->0,

   {"c2","passDirect","mask"}->0,
   {"c2","change","mask"}->0,
   {"c2","changed","mask"}->0,
   {"c2","jacobian2","mask"}->0



},LossFunction->"Loss"
];


ReverseCouplingLayer[net_,z_,mask_]:=(
   mu=NetTake[net,"mu"][z];
   scale=NetTake[net,"scale"][z];
   ch=z*scale+mu;
   tot=ch*(1-mask)+z*mask)


ReverseRealNVP[ z_ ] := (
   oc2i = ReverseCouplingLayer[ NetExtract[ train, {"c2"} ], z, 1-channelmask64x1x1 ];
   oc1i = ReverseCouplingLayer[ NetExtract[ train, {"c1"} ], oc2i, channelmask64x1x1 ];
   reshape = ReshapeLayer[{4,4,4}][oc1i];
   ReverseRealNVPBlock[ reshape ]
)


ReverseSqueezeLayer[ inp_ ] := (
   tmp = ConstantArray[0,{1,8,8}];
   tmp[[1,1;;8;;2,1;;8;;2]] = inp[[1]];
   tmp[[1,1;;8;;2,2;;8;;2]] = inp[[2]];
   tmp[[1,2;;8;;2,1;;8;;2]] = inp[[3]];
   tmp[[1,2;;8;;2,2;;8;;2]] = inp[[4]];
   tmp
);


ReverseRealNVPBlock[ z_ ] := (
   c4i = ReverseCouplingLayer[ NetExtract[ train, {"block","c4"} ], z, 1 - channelmask4x4x4 ];
   c3i = ReverseCouplingLayer[ NetExtract[ train, {"block","c3"} ], c4i, channelmask4x4x4 ];
   desq = ReverseSqueezeLayer[ c3i ];
   c2i = ReverseCouplingLayer[ NetExtract[ train, {"block","c2"} ], desq, 1 - checkerboard1x8x8 ];
   c1i = ReverseCouplingLayer[ NetExtract[ train, {"block","c1"} ], c2i, checkerboard1x8x8 ]
)


gaussiansample[] := Table[RandomVariate@NormalDistribution[0,1],{64},{1},{1}]


sample[] := ReverseRealNVP[ gaussiansample[] ];
