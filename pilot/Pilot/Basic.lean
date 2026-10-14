import Std

namespace Pilot

abbrev Bits (n : Nat) := Fin n → Bool

def setBit {n : Nat} (x : Bits n) (i : Fin n) (b : Bool) : Bits n :=
  fun j => if j = i then b else x j

def flip {n : Nat} (x : Bits n) (i : Fin n) : Bits n :=
  setBit x i (!(x i))

@[simp] theorem flip_flip {n : Nat} (x : Bits n) (i : Fin n) :
    flip (flip x i) i = x := by
  funext j
  by_cases h : j = i
  · subst j; simp [flip, setBit]
  · simp [flip, setBit, h]

theorem flip_injective {n : Nat} (i : Fin n) {a b : Bits n}
    (h : flip a i = flip b i) : a = b := by
  have hh := congrArg (fun x => flip x i) h
  simpa using hh

-- This matches the upstream Python definition, not a real-valued derivative.
def derivative {n : Nat} (f : Bits n → Bool) (x : Bits n) (i : Fin n) : Bool :=
  Bool.xor (f (setBit x i false)) (f (setBit x i true))

theorem derivative_flip_form {n : Nat} (f : Bits n → Bool)
    (x : Bits n) (i : Fin n) :
    derivative f x i = Bool.xor (f x) (f (flip x i)) := by
  have same : setBit x i (x i) = x := by
    funext j; simp [setBit]; split <;> simp_all
  cases h : x i
  · have h0 : setBit x i false = x := by simpa [h] using same
    simp [derivative, flip, h, h0]
  · have h1 : setBit x i true = x := by simpa [h] using same
    simp [derivative, flip, h, h1, Bool.xor_comm]

@[simp] theorem derivative_flip {n : Nat} (f : Bits n → Bool)
    (x : Bits n) (i : Fin n) : derivative f (flip x i) i = derivative f x i := by
  rw [derivative_flip_form, derivative_flip_form]
  simp [Bool.xor_comm]

def ccz (a b c : Bool) : Bool := a && b && c

theorem ccz_d0 (a b c : Bool) :
    Bool.xor (ccz false b c) (ccz true b c) = (b && c) := by
  cases a <;> cases b <;> cases c <;> decide

theorem ccz_d1 (a b c : Bool) :
    Bool.xor (ccz a false c) (ccz a true c) = (a && c) := by
  cases a <;> cases b <;> cases c <;> decide

theorem ccz_d2 (a b c : Bool) :
    Bool.xor (ccz a b false) (ccz a b true) = (a && b) := by
  cases a <;> cases b <;> cases c <;> decide

def cczPhase (x : Bits 3) : Bool := ccz (x 0) (x 1) (x 2)

theorem ccz_derivative0 (x : Bits 3) : derivative cczPhase x 0 = (x 1 && x 2) := by
  simpa [derivative, cczPhase, setBit] using ccz_d0 (x 0) (x 1) (x 2)

theorem ccz_derivative1 (x : Bits 3) : derivative cczPhase x 1 = (x 0 && x 2) := by
  simpa [derivative, cczPhase, setBit] using ccz_d1 (x 0) (x 1) (x 2)

theorem ccz_derivative2 (x : Bits 3) : derivative cczPhase x 2 = (x 0 && x 1) := by
  simpa [derivative, cczPhase, setBit] using ccz_d2 (x 0) (x 1) (x 2)

-- Explicit refutations: rejection by a prover alone does not prove falsity.
theorem ccz_wrong_omits_factor :
    ¬ (∀ a b c : Bool, Bool.xor (ccz false b c) (ccz true b c) = b) := by
  intro h
  have bad := h false true false
  contradiction

theorem ccz_wrong_zero_derivative :
    ¬ (∀ a b c : Bool, Bool.xor (ccz false b c) (ccz true b c) = false) := by
  intro h
  have bad := h false true true
  contradiction

end Pilot
