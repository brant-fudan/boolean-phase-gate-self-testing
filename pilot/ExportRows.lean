import Pilot
open Pilot

def bitsOf (n k : Nat) : Bits n := fun i => k.testBit (n - 1 - i.val)
def indexOf {n : Nat} (x : Bits n) : Nat :=
  (List.finRange n).foldl (fun a i => 2*a + if x i then 1 else 0) 0

def main : IO Unit := do
  let out ← IO.getStdout
  out.putStrLn "n,table,site,a,b,derivative,amplitude,z_image,x_image"
  for n in [1, 2, 3] do
    let d := 2^n
    for t in List.range (2^d) do
      let f : Bits n → Bool := fun x => t.testBit (indexOf x)
      for i in List.finRange n do
        for a in List.range d do
          for b in List.range d do
            let x := bitsOf n a
            let y := bitsOf n b
            let der := if derivative f y i then 1 else 0
            out.putStrLn s!"{n},{t},{i.val},{a},{b},{der},{choi f x y},{stabilizerZ i (choi f) x y},{stabilizerX f i (choi f) x y}"
