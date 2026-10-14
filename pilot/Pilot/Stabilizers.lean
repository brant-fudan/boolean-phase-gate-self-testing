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

end Pilot
