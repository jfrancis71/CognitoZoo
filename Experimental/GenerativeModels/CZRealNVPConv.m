(* ::Package:: *)

gauss = NetChain[{
   ElementwiseLayer[-0.5*#^2-Log[Sqrt[2*Pi]]&],\
   SummationLayer[]}];


MaskLayer[ mask_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


checkerboard = ConstantArray[0,{1,8,8}];
checkerboard[[1,1;;8;;2,1;;8;;2]] = 1;
checkerboard[[1,2;;8;;2,2;;8;;2]] = 1;


StableScaleNet = NetGraph[{"nonlinearity"->Tanh,"c1"->ConstantArrayLayer[{1}],"c2"->ConstantArrayLayer[{1}],"times"->ThreadingLayer[Times],"plus"->ThreadingLayer[Plus]},{{"nonlinearity","c1"}->"times",{"times","c2"}->"plus"}];


CouplingLayer[ next_, passThrough_ ] := NetGraph[{
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
   "next"->next,
   "jacobian1"->ElementwiseLayer[-#&],
   "jacobian2"->MaskLayer[1-passThrough],
   "jacobian3"->SummationLayer[],
   "logdensity"->ThreadingLayer[Plus]
},{
   "passDirect"->{"mu","logscale"},
   "logscale"->"scale"->"invscale",
   "mu"->"invmu",
   {"change","invmu"}->"inv1",
   {"inv1","invscale"}->"inv2"->"changed",
   {"changed","passDirect"}->"new"->"next",
   "logscale"->"jacobian1"->"jacobian2"->"jacobian3",
   {"jacobian3","next"}->"logdensity"
}]


channelmask = ConstantArray[0,{64,1,1}];
channelmask[[1;;32,1,1]] = 1;


back = NetChain[{
   ReshapeLayer[{64,1,1}],
   CouplingLayer[CouplingLayer[gauss,channelmask],1-channelmask]}];


back


RealNVP = NetGraph[{
   CouplingLayer[ back, checkerboard ],
   ElementwiseLayer[-#&]},{1->2->NetPort["Loss"]}];


images = ResourceData["MNIST","TrainingData"][[5924;;12665,1]];


resize = ImageResize[#,{8,8}]&/@images;


data = {ImageData[#]}&/@resize;data//Dimensions;


data1 = data+Table[RandomReal[]/10,{6742},{1},{8},{8}];


train = NetTrain[ RealNVP, Association["Input"->#]&/@data1,LearningRateMultipliers->{
   {1,2,"passDirect","mask"}->0,
   {1,2,"change","mask"}->0,
   {1,2,"changed","mask"}->0,
   {1,2,"jacobian2","mask"}->0,

   {1,2,"next","passDirect","mask"}->0,
   {1,2,"next","change","mask"}->0,
   {1,2,"next","changed","mask"}->0,
   {1,2,"next","jacobian2","mask"}->0

},LossFunction->"Loss"
]


reverse[net_,z_,mask_]:=(
   mu=NetTake[net,"mu"][z];
   scale=NetTake[net,"scale"][z];
   ch=z*scale+mu;
   tot=ch*(1-mask)+z*mask)


sampleFromGaussian[ z_ ] := (
   z2 = reverse[ NetExtract[ train, {1,"next",2,"next"} ], z, channelmask ];
   z1 = reverse[ NetExtract[ train, {1,"next",2} ], z2, 1-channelmask ];
   rz1 = ReshapeLayer[{1,8,8}][z1];
   z0 = reverse[ NetExtract[ train, {1} ], rz1, checkerboard ]
)


gaussiansample[] := Table[RandomVariate@NormalDistribution[0,1],{64},{1},{1}]
