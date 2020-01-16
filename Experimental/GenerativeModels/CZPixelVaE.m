(* ::Package:: *)

<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"
<<"Experimental/GenerativeModels/CZPixelCNN.m"


PixelVaEEncoderBinary[ imageDims_, latentUnits_ ] := NetChain[{
   FlattenLayer[],
   CZCreateEncoder[ imageDims[[1]]*imageDims[[2]], latentUnits ]}];


PixelVaEEncoderRealGauss[ imageDims_, latentUnits_ ] := NetChain[{
   FlattenLayer[],
   CZCreateEncoder[ imageDims[[1]]*imageDims[[2]], latentUnits ]}];


PixelVaEEncoderDiscrete[ imageDims_, latentUnits_ ] := NetChain[{
   FlattenLayer[],
   CZCreateEncoder[ imageDims[[1]]*imageDims[[2]]*10, latentUnits ]}];


CZPixelVaEDecoder[ crossEntropyType_, imageDims_ ] := NetGraph[{
   {500,Ramp},
   {500,Ramp},
   {imageDims[[1]]*imageDims[[2]]},
   ReshapeLayer[imageDims],
   CZCreatePixelCNNConditionalNet[ crossEntropyType, PixelCNNOrdering[ imageDims ] ]},{
   NetPort["Conditional"]->1->2->3->4->NetPort[{5,"Conditional"}],
   NetPort["Target"]->NetPort[{5,"Input"}]}];


CZCreatePixelVaEBinary[ imageDims_:{28,28}, latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   CZBinary[imageDims],
   Identity,
   CZCreateVaENet[ PixelVaEEncoderBinary[ imageDims, latentUnits ], CZPixelVaEDecoder[ CZBinary[{28,28}], imageDims ] ]];


CZCreatePixelVaEDiscrete[ imageDims_:{28,28}, latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   CZDiscrete[imageDims],
   CZOneHot,
   CZCreateVaENet[ PixelVaEEncoderDiscrete[ imageDims, latentUnits ], CZPixelVaEDecoder[ CZDiscrete[{28,28}], imageDims ] ]];


SyntaxInformation[ CZPixelVaE ]= {"ArgumentsPattern"->{_}};


CZSample[ CZGenerativeModel[ CZPixelVaE[ latentUnits_ ], inputType_, encoder_, pixelCNNNet_ ] ] := (
   z = CZSampleVaELatent[ latentUnits ];
   cond = NetTake[ NetExtract[ pixelCNNNet, "decoder" ],{1,4} ][ z ];
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, {"decoder",5} ], inputType, encoder, cond ]
)


CZCreatePixelVaERealGauss[ imageDims_:{28,28}, latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   CZRealGauss[imageDims],
   Identity,
   CZCreateVaENet[ PixelVaEEncoderRealGauss[ imageDims, latentUnits ], CZPixelVaEDecoder[ CZRealGauss[{28,28}], imageDims ] ]];
