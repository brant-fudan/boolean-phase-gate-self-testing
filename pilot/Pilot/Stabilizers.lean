import Pilot.Basic

namespace Pilot

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

-- The fixed vector is genuinely nonzero, so +1 is an eigenvalue.
theorem choi_nonzero {n : Nat} (f : Bits n → Bool) :
    choi f ≠ (fun _ _ => 0) := by
  intro h
  let x : Bits n := fun _ => false
  have hz := congrFun (congrFun h x) x
  have hs : sign (f x) = 0 := by simpa [choi] using hz
  cases hx : f x <;> simp [sign, hx] at hs

end Pilot
