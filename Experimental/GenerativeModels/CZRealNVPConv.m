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


channelmask = ConstantArray[0,{64,1,1}];
channelmask[[1;;32,1,1]] = 1;


RealNVP = NetGraph[{
   CouplingLayer[ checkerboard ],
   CouplingLayer[ 1 - checkerboard ],
   ReshapeLayer[{64,1,1}],
   CouplingLayer[ channelmask ],
   CouplingLayer[ 1 - channelmask ],
   gauss,
   ThreadingLayer[Plus],
   ElementwiseLayer[ -#& ]
},{
   NetPort[{1,"Transformed"}]->2,
   NetPort[{2,"Transformed"}]->3,
   3->4,
   NetPort[{4,"Transformed"}]->5,
   NetPort[{5,"Transformed"}]->6,
   {NetPort[{1,"Jacobian"}],NetPort[{2,"Jacobian"}],NetPort[{4,"Jacobian"}],NetPort[{5,"Jacobian"}],6}
      ->7->8->NetPort["Loss"]
},"Input"->{1,8,8}];


images = ResourceData["MNIST","TrainingData"][[5924;;12665,1]];


resize = ImageResize[#,{8,8}]&/@images;


data = {ImageData[#]}&/@resize;data//Dimensions;


data1 = data+Table[RandomReal[]/10,{6742},{1},{8},{8}];


train = NetTrain[ RealNVP, Association["Input"->#]&/@data1,LearningRateMultipliers->{

   {1,"passDirect","mask"}->0,
   {1,"change","mask"}->0,
   {1,"changed","mask"}->0,
   {1,"jacobian2","mask"}->0,
   
   {2,"passDirect","mask"}->0,
   {2,"change","mask"}->0,
   {2,"changed","mask"}->0,
   {2,"jacobian2","mask"}->0,
   
   {4,"passDirect","mask"}->0,
   {4,"change","mask"}->0,
   {4,"changed","mask"}->0,
   {4,"jacobian2","mask"}->0,
   
   {5,"passDirect","mask"}->0,
   {5,"change","mask"}->0,
   {5,"changed","mask"}->0,
   {5,"jacobian2","mask"}->0


},LossFunction->"Loss"
];


reverse[net_,z_,mask_]:=(
   mu=NetTake[net,"mu"][z];
   scale=NetTake[net,"scale"][z];
   ch=z*scale+mu;
   tot=ch*(1-mask)+z*mask)


sampleFromGaussian[ z_ ] := (
   z3 = reverse[ NetExtract[ train, {5} ], z, 1-channelmask ];
   z2 = reverse[ NetExtract[ train, {4} ], z3, channelmask ];
   rz2 = ReshapeLayer[{1,8,8}][z2];
   z1 = reverse[ NetExtract[ train, {2} ], rz2, 1-checkerboard ];
   z0 = reverse[ NetExtract[ train, {1} ], z1, checkerboard ]
)


gaussiansample[] := Table[RandomVariate@NormalDistribution[0,1],{64},{1},{1}]
