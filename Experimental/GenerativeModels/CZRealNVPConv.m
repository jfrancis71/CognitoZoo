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


checkerboard4x8x8 = ConstantArray[0,{4,8,8}];
checkerboard4x8x8[[All,1;;8;;2,1;;8;;2]] = 1;
checkerboard4x8x8[[All,2;;8;;2,2;;8;;2]] = 1;


checkerboard1x16x16 = ConstantArray[0,{1,16,16}];
checkerboard1x16x16[[1,1;;16;;2,1;;16;;2]] = 1;
checkerboard1x16x16[[1,2;;16;;2,2;;16;;2]] = 1;


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


channelmask16x4x4 = ConstantArray[0,{16,4,4}];
channelmask16x4x4[[1;;8,All,All]] = 1;


channelmask4x8x8 = ConstantArray[0,{4,8,8}];
channelmask4x8x8[[1;;2,All,All]] = 1;


channelmask64x1x1 = ConstantArray[0,{64,1,1}];
channelmask64x1x1[[1;;32,1,1]] = 1;


channelmask256x1x1 = ConstantArray[0,{256,1,1}];
channelmask256x1x1[[1;;64,1,1]] = 1;


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


RealNVPBlock1 = NetGraph[{
   "c1"->CouplingLayer[ checkerboard1x16x16 ],
   "c2"->CouplingLayer[ 1 - checkerboard1x16x16 ],
   "s1"->SqueezeLayer2x2[ 1 ],
   "c3"->CouplingLayer[ channelmask4x8x8 ],
   "c4"->CouplingLayer[ 1 - channelmask4x8x8 ],
   "jacobian"->ThreadingLayer[Plus]},{
   NetPort[{"c1","Transformed"}]->"c2",
   NetPort[{"c2","Transformed"}]->"s1",
   "s1"->"c3",
   NetPort[{"c3","Transformed"}]->"c4",
   NetPort[{"c4","Transformed"}]->NetPort["Transformed"],
   {NetPort["c1","Jacobian"],NetPort["c2","Jacobian"],NetPort["c3","Jacobian"],NetPort["c4","Jacobian"]}->
      "jacobian"->NetPort["Jacobian"]
   }];


RealNVPBlock2 = NetGraph[{
   "c1"->CouplingLayer[ checkerboard4x8x8 ],
   "c2"->CouplingLayer[ 1 - checkerboard4x8x8 ],
   "s1"->SqueezeLayer2x2[ 4 ],
   "c3"->CouplingLayer[ channelmask16x4x4 ],
   "c4"->CouplingLayer[ 1 - channelmask16x4x4 ],
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
   "block1"->RealNVPBlock1,
   "block2"->RealNVPBlock2,
   "reshape"->ReshapeLayer[{256,1,1}],
   "c1"->CouplingLayer[ channelmask256x1x1 ],
   "c2"->CouplingLayer[ 1 - channelmask256x1x1 ],
   "g"->gauss,
   "th"->ThreadingLayer[Plus],
   "loss"->ElementwiseLayer[ -#& ]
},{
   NetPort[{"block1","Transformed"}]->"block2",
   NetPort[{"block2","Transformed"}]->"reshape"->"c1",
   NetPort[{"c1","Transformed"}]->"c2",
   NetPort[{"c2","Transformed"}]->"g",
   {NetPort[{"block1","Jacobian"}],NetPort[{"block2","Jacobian"}],"g",NetPort[{"c1","Jacobian"}],NetPort[{"c2","Jacobian"}]}->"th"->"loss"->NetPort["Loss"]
},"Input"->{1,16,16}];


images = ResourceData["MNIST","TrainingData"][[5924;;12665,1]];


resize = ImageResize[#,{16,16}]&/@images;


data = {ImageData[#]}&/@resize;data//Dimensions;


data1 = data+Table[RandomReal[]/10,{6742},{1},{16},{16}];


train = NetTrain[ RealNVP, Association["Input"->#]&/@data1,LearningRateMultipliers->{
   {"block1","s1","s1"}->0,
   {"block1","s1","s2"}->0,
   {"block1","s1","s3"}->0,
   {"block1","s1","s4"}->0,


   {"block1","c1","passDirect","mask"}->0,
   {"block1","c1","change","mask"}->0,
   {"block1","c1","changed","mask"}->0,
   {"block1","c1","jacobian2","mask"}->0,

   {"block1","c2","passDirect","mask"}->0,
   {"block1","c2","change","mask"}->0,
   {"block1","c2","changed","mask"}->0,
   {"block1","c2","jacobian2","mask"}->0,
   
   {"block1","c3","passDirect","mask"}->0,
   {"block1","c3","change","mask"}->0,
   {"block1","c3","changed","mask"}->0,
   {"block1","c3","jacobian2","mask"}->0,
   
   {"block1","c4","passDirect","mask"}->0,
   {"block1","c4","change","mask"}->0,
   {"block1","c4","changed","mask"}->0,
   {"block1","c4","jacobian2","mask"}->0,

   {"block2","s1","s1"}->0,
   {"block2","s1","s2"}->0,
   {"block2","s1","s3"}->0,
   {"block2","s1","s4"}->0,


   {"block2","c1","passDirect","mask"}->0,
   {"block2","c1","change","mask"}->0,
   {"block2","c1","changed","mask"}->0,
   {"block2","c1","jacobian2","mask"}->0,

   {"block2","c2","passDirect","mask"}->0,
   {"block2","c2","change","mask"}->0,
   {"block2","c2","changed","mask"}->0,
   {"block2","c2","jacobian2","mask"}->0,
   
   {"block2","c3","passDirect","mask"}->0,
   {"block2","c3","change","mask"}->0,
   {"block2","c3","changed","mask"}->0,
   {"block2","c3","jacobian2","mask"}->0,
   
   {"block2","c4","passDirect","mask"}->0,
   {"block2","c4","change","mask"}->0,
   {"block2","c4","changed","mask"}->0,
   {"block2","c4","jacobian2","mask"}->0,


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
   oc2i = ReverseCouplingLayer[ NetExtract[ train, {"c2"} ], z, 1-channelmask256x1x1 ];
   oc1i = ReverseCouplingLayer[ NetExtract[ train, {"c1"} ], oc2i, channelmask256x1x1 ];
   reshape = ReshapeLayer[{16,4,4}][oc1i];
   revblock2 = ReverseRealNVPBlock2[ reshape ];
   revblock1 = ReverseRealNVPBlock1[ revblock2 ]
)


ReverseSqueezeLayer1[ inp_ ] := (ky=inp;
   tmp = ConstantArray[0,{1,16,16}];
   tmp[[1,1;;16;;2,1;;16;;2]] = inp[[1]];
   tmp[[1,1;;16;;2,2;;16;;2]] = inp[[2]];
   tmp[[1,2;;16;;2,1;;16;;2]] = inp[[3]];
   tmp[[1,2;;16;;2,2;;16;;2]] = inp[[4]];
   tmp
);


ReverseSqueezeLayer2[ inp_ ] := (
   tmp = ConstantArray[0,{4,8,8}];
   tmp[[All,1;;8;;2,1;;8;;2]] = inp[[1;;4]];
   tmp[[All,1;;8;;2,2;;8;;2]] = inp[[5;;8]];
   tmp[[All,2;;8;;2,1;;8;;2]] = inp[[9;;12]];
   tmp[[All,2;;8;;2,2;;8;;2]] = inp[[13;;16]];
   tmp
);


ReverseRealNVPBlock1[ z_ ] := (
   c4i = ReverseCouplingLayer[ NetExtract[ train, {"block1","c4"} ], z, 1 - channelmask4x8x8 ];
   c3i = ReverseCouplingLayer[ NetExtract[ train, {"block1","c3"} ], c4i, channelmask4x8x8 ];tp6=c3i;
   desq1 = ReverseSqueezeLayer1[ c3i ];
   c2i1 = ReverseCouplingLayer[ NetExtract[ train, {"block1","c2"} ], desq1, 1 - checkerboard1x16x16 ];
   c1i2 = ReverseCouplingLayer[ NetExtract[ train, {"block1","c1"} ], c2i1, checkerboard1x16x16 ]
)


ReverseRealNVPBlock2[ z_ ] := (
   c4i = ReverseCouplingLayer[ NetExtract[ train, {"block2","c4"} ], z, 1 - channelmask16x4x4 ];
   c3i = ReverseCouplingLayer[ NetExtract[ train, {"block2","c3"} ], c4i, channelmask16x4x4 ];
   desq = ReverseSqueezeLayer2[ c3i ];
   c2i = ReverseCouplingLayer[ NetExtract[ train, {"block2","c2"} ], desq, 1 - checkerboard4x8x8 ];mytmp1=c2i;
   c1i = ReverseCouplingLayer[ NetExtract[ train, {"block2","c1"} ], c2i, checkerboard4x8x8 ]
)


gaussiansample[] := Table[RandomVariate@NormalDistribution[0,1],{256},{1},{1}]


sample[] := ReverseRealNVP[ gaussiansample[] ];
