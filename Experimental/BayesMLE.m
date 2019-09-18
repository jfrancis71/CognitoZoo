(* ::Package:: *)

(*
   Note these are all discrete distributions. Do not try and pass a continuous
   distribution, eg from a pixel in.
*)


CZSoftmax[v_]:=Exp[v]/Total[Exp[v]]


(*
   wt is a weight vector
*)
CZBayesProbT[ t_, wt_ ] := Exp[wt[[t]]]/Total[Exp[wt]];


(*
   pcond[v,t,w] v is a vector, t is a scalar between 1...N, w is a tensor N*|v|*U where U is number of possible entries in single element of v
 returns a scalar
*)
CZBayesCondProbVonT[ v_, t_, w_ ] := Apply[Times,MapThread[#1[[1+#2]]&,{(CZSoftmax[#]&/@w[[t]]),v}]]


(*
   jt[v,t,{w,wt}] is the joint distribution for (v,t) where w is conditional prob matrix and wt is vector prob matrix
*)
CZBayesJointT[ v_, t_, {w_,wt_}] := CZBayesCondProbVonT[v,t,w]*CZBayesProbT[t,wt];


(*
   v[v,w,nt] is marginal distribution over v given structure w which is {w,wt}
*)
CZBayesMarginalV[ v_, w_, nt_] := Sum[CZBayesJointT[v,t,w],{t,1,nt}];


(*
   l[v,w,nt] is loss nt is number of entries in t.
*)
CZBayesLoss[ v_, w_, nt_] := Log[CZBayesMarginalV[v,w,nt]];


(*
   dat=Join[Tuples[{0,1},2],ConstantArray[{1,1},8]];
   BayesMLE[dat,2]
   returns {-10.043858601430028`,{w[1,1]\[Rule]-0.05589845026206474`,w[1,2]\[Rule]-0.0548370752934956`,w[2,1]\[Rule]5.016771454675282`,w[2,2]\[Rule]4.998251823564115`,wt[1]\[Rule]0.5286574893407507`,wt[2]\[Rule]1.3031577035237136`}}
*)


(*
   data is in Examples*Features format, N is number of hidden states
*)
CZBayesMLE[data_,N_,U_] := Module[{fw,fwt,loss,tw1},
   fw=Table[w[t,k,u],{t,1,N},{k,1,Length[data[[1]]]},{u,1,U}];de=fw;
   fwt=Table[wt[t],{t,1,N}];fw1=fw;fwt1=fwt;
   loss=Total[CZBayesLoss[#,{fw,fwt},N]&/@data];floss=loss;
   tw1=Join[
      Flatten[Table[{w[t,k,u],Random[]},{t,1,N},{k,1,Length[data[[1]]]},{u,1,U}],2],
      Table[{wt[t],Random[]},{t,1,N}]
   ];tw11=tw1;
   algo = FindMaximum[loss,tw1];zalgo=algo;
   {algo[[1]]/Length[data], Table[w[t,k,u],{t,1,N},{k,1,Length[data[[1]]]},{u,1,U}] /. algo[[2]]}
];


(* Neural net implementation not working great. Not sure why
*)


CreateCondNet:=NetGraph[{
   {ConstantArrayLayer["Output"->100,"Array"->RandomReal[{-.5,.5},100]],LogisticSigmoid},
   ThreadingLayer[Times],
   ElementwiseLayer[1-#&],
   ElementwiseLayer[1-#&],
   ThreadingLayer[Times],
   ThreadingLayer[Plus],
   ElementwiseLayer[Log[#]&],
   SummationLayer[]
  },{
   {1,NetPort["Input"]}->2,
   1->4,
   {3,4}->5,
   {2,5}->6->7->8
}];


CreateBayesNet[k_]:=NetGraph[Flatten@Join[{
   "prior"->{ConstantArrayLayer["Output"->k,"Array"->RandomReal[{-.001,.001},k]],SoftmaxLayer[]}},
   {Table[ToString[n]->{CreateCondNet,ElementwiseLayer[Exp],ReplicateLayer[1,"Input"->"Real"]},{n,1,k}]},
   {"concat"->CatenateLayer["Inputs"->ConstantArray[1,k]],
   "post"->ThreadingLayer[Times],
   "plus"->SummationLayer[],
   "loss"->ElementwiseLayer[-Log[#]&]}],{
   Table[ToString[n],{n,k}]->"concat",
   {"concat","prior"}->"post"->"plus",
   "plus"->"loss"->NetPort["Loss"]
}]

