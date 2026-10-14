import Pilot.Basic

namespace Pilot

def sign (b : Bool) : Int := if b then -1 else 1

@[simp] theorem sign_square (b : Bool) : sign b * sign b = 1 := by
  cases b <;> decide

theorem sign_cancel_left (p q : Bool) :
    sign (Bool.xor p q) * sign p = sign q := by
  cases p <;> cases q <;> decide

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

theorem choi_stabilizerZ {n : Nat} (f : Bits n → Bool) (i : Fin n) :
    stabilizerZ i (choi f) = choi f := by
  funext a b
  by_cases h : a = b
  · subst b; simp [stabilizerZ, choi]
  · simp [stabilizerZ, choi, h]

theorem choi_stabilizerX {n : Nat} (f : Bits n → Bool) (i : Fin n) :
    stabilizerX f i (choi f) = choi f := by
  funext a b
  by_cases h : a = b
  · subst b
    simp only [stabilizerX, choi, if_pos rfl, derivative_flip_form, flip_flip]
    exact sign_cancel_left (f (flip a i)) (f a)
  · have hf : flip a i ≠ flip b i := fun hh => h (flip_injective i hh)
    simp [stabilizerX, choi, h, hf]

end Pilot
