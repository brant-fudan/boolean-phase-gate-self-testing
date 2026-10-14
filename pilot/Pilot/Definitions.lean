import Std

namespace Pilot

abbrev Bits (n : Nat) := Fin n → Bool

instance {n : Nat} : DecidableEq (Bits n) := fun a b =>
  decidable_of_iff (∀ i : Fin n, a i = b i) ⟨fun h => funext h, fun h => congrFun h⟩

def setBit {n : Nat} (x : Bits n) (i : Fin n) (b : Bool) : Bits n :=
  fun j => if j = i then b else x j

def flip {n : Nat} (x : Bits n) (i : Fin n) : Bits n :=
  setBit x i (!(x i))

-- This matches the upstream Python definition, not a real-valued derivative.
def derivative {n : Nat} (f : Bits n → Bool) (x : Bits n) (i : Fin n) : Bool :=
  Bool.xor (f (setBit x i false)) (f (setBit x i true))

def ccz (a b c : Bool) : Bool := a && b && c

def cczPhase (x : Bits 3) : Bool := ccz (x 0) (x 1) (x 2)

def sign (b : Bool) : Int := if b then -1 else 1

-- Exact, unnormalized computational-basis amplitudes.
-- No floating-point tolerance, Hilbert-space self-testing, or normalization claim.
def choi {n : Nat} (f : Bits n → Bool) (a b : Bits n) : Int :=
  if a = b then sign (f a) else 0

def stabilizerZ {n : Nat} (i : Fin n) (v : Bits n → Bits n → Int)
    (a b : Bits n) : Int := sign (a i) * sign (b i) * v a b

-- Pullback at the preimage: X_A X_B D_B, with D applied first.
def stabilizerX {n : Nat} (f : Bits n → Bool) (i : Fin n)
    (v : Bits n → Bits n → Int) (a b : Bits n) : Int :=
  sign (derivative f (flip b i) i) * v (flip a i) (flip b i)

end Pilot
