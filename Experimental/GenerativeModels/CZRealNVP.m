(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


gauss = NetChain[{ElementwiseLayer[-0.5*#^2 - Log[Sqrt[2*Pi]]&]},"Input"->"Real","Output"->"Real"];


gauss2D = NetGraph[{
   "p1"->{PartLayer[1],gauss},
   "p2"->{PartLayer[2],gauss},
   "sum"->TotalLayer[]},{
   {"p1","p2"}->"sum"},
   "Input"->{2}];


StableScaleNet = NetGraph[{
   "nonlinearity"->Tanh,
   "c1"->ConstantArrayLayer[{1}],
   "c2"->ConstantArrayLayer[{1}],
   "times"->ThreadingLayer[Times],
   "plus"->ThreadingLayer[Plus]},{
   {"nonlinearity","c1"}->"times",
   {"times","c2"}->"plus"}];


(* scale and shift second element *)
RealNVPCouplingLayer[ next_ ] := NetGraph[{
   "p1"->PartLayer[1;;1],
   "p2"->PartLayer[2;;2],
   "mu"->{16,Tanh,16,Tanh,16,Tanh,1},
   "logscale"->{16,Tanh,16,Tanh,16,Tanh,1,StableScaleNet},
   "scalarlogscale"->{PartLayer[1],ElementwiseLayer[-#&]},
   "scale"->ElementwiseLayer[Exp],
   "invmu"->ElementwiseLayer[-#&],
   "invscale"->ElementwiseLayer[1/#&],
   "inv1"->ThreadingLayer[Plus],
   "inv2"->ThreadingLayer[Times],
   "cat"->CatenateLayer[],
   "gauss"->next,
   "res"->ThreadingLayer[Plus]},{
   "p1"->{"mu","logscale"},
   "mu"->"invmu",
   "logscale"->"scale"->"invscale",
   {"p2","invmu"}->"inv1",
   {"inv1","invscale"}->"inv2",
   {"p1","inv2"}->"cat"->"gauss",
   "logscale"->"scalarlogscale",
   {"gauss","scalarlogscale"}->"res"
   },
   "Input"->{2}]


RealNVPPermutationLayer[ layer_ ] := NetGraph[{
   "p1"->PartLayer[1;;1],
   "p2"->PartLayer[2;;2],
   "cat"->CatenateLayer[],
   "next"->layer},{
   {"p2","p1"}->"cat"->"next"},
   "Input"->{2}]


RealNVP = NetGraph[{
   "coupling"->RealNVPCouplingLayer@RealNVPPermutationLayer@RealNVPCouplingLayer@RealNVPPermutationLayer[ RealNVPCouplingLayer[ gauss2D ] ],
   "loss"->ElementwiseLayer[-#&]},{
   "coupling"->"loss"->{NetPort["Loss"],NetPort["Dummy"]}
(* Our dummy port is in their because other parts of this framework will pass in "Input" and ask for "Loss"
   but if there is only one output port, it is confused as to the meaning of port "Loss". So I just want a second
   output port.
*)
}];


SyntaxInformation[ CZRealNVP ]= {"ArgumentsPattern"->{}};


CZCreateRealNVP[] := CZGenerativeModel[
   CZRealNVP,
   CZRealVector[ 2 ],
   Identity,
   RealNVP ]
