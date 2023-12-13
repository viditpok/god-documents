/**
*@author Vidit Pokharna
*@version 1.0
*/
public class Cat extends Pet implements Treatable {
    /**
     *
     */
    protected boolean hasStripes;

    /**
     * @param name string
     * @param age int
     * @param painLevel int
     * @param hasStripes boolean
     */
    public Cat(String name, int age, int painLevel, boolean hasStripes) {
        super(name, age, painLevel);
        this.hasStripes = hasStripes;
    }

    /**
     * @param hasStripes boolean
     */
    public Cat(boolean hasStripes) {
        this("Purrfect", 4, 9, hasStripes);
    }

    /**
     * @param Pet p
     */
    @Override
    public void playWith(Pet p) {
        int pain = super.getPainLevel();
        if (p instanceof Cat) {
            Cat c = (Cat) p;
            if (hasStripes == c.hasStripes) {
                if (pain < 5) {
                    super.setPainLevel(1);
                } else {
                    super.setPainLevel(pain - 4);
                }
                System.out.println("Meow! I love playing with other cats with the same pattern as me");
            } else {
                if (pain < 3) {
                    super.setPainLevel(1);
                } else {
                    super.setPainLevel(pain - 2);
                }
                System.out.println("Meow! I like playing with other cats without the same pattern as me");
            }
        } else if (p instanceof Dog) {
            if (pain < 10) {
                super.setPainLevel(pain + 1);
            } else {
                super.setPainLevel(10);
            }
            System.out.println("Meow. Go away " + p.getName() + "! I don’t like playing with Dogs!");
        }
    }

    /**
     *
     */
    @Override
    public void treat() {
        int pain = super.getPainLevel();
        if (pain > 1) {
            super.setPainLevel(pain - 1);
        }
    }

    /**
     * @return String
     */
    public String toString() {
        return (super.toString() + "My age in human years is "
            + Treatable.convertCatToHumanYears(super.getAge()) + ".");
    }

    /**
     * @param Object obj
     * @return boolean
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Cat)) {
            return false;
        }
        Cat other = (Cat) obj;
        return (name.equals(other.getName()) && age == other.getAge()
            && painLevel == other.getPainLevel() && hasStripes == other.isHasStripes());
    }

    /**
     * @return boolean
     */
    public boolean isHasStripes() {
        return hasStripes;
    }

    /**
     * @param hasStripes boolean
     */
    public void setHasStripes(boolean hasStripes) {
        this.hasStripes = hasStripes;
    }
}