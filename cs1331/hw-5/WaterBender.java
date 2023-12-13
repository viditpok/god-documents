/**
*@author Vidit Pokharna
*@version 1.0
*/
public class WaterBender extends Bender {
    /**
     *
     */
    private boolean healer;

    /**
     * @param name string
     * @param strength int
     * @param health int
     * @param healer boolean
     */
    public WaterBender(String name, int strength, int health, boolean healer) {
        super(name, strength, health);
        this.healer = healer;
    }

    /**
     * @param name string
     */
    public WaterBender(String name) {
        this(name, 40, 80, false);
    }


    /**
     * @param b bender
     */
    public void attack(Bender b) {
        if (this.isAlive()) {
            if (b instanceof FireBender) {
                int newHealth = b.getHealth() - strength;
                if (newHealth < 0) {
                    newHealth = 0;
                }
                b.setHealth(newHealth);
            } else if (b instanceof WaterBender) {
                int newHealth = b.getHealth() - 1;
                if (newHealth < 0) {
                    newHealth = 0;
                }
                b.setHealth(newHealth);
            }
        }
    }

    /**
     * @param wb waterbender
     */
    public void heal(WaterBender wb) {
        if (!wb.isHealer() || !wb.isAlive()) {
            return;
        } else if (wb.isHealer()) {
            wb.setHealth(wb.getHealth() + 40);
        } else if (!(wb.isHealer())) {
            wb.setHealth(wb.getHealth() + 20);
        }
    }

    /**
     * @return string
     */
    public String toString() {
        String c = "";
        if (healer) {
            c = "can";
        } else {
            c = "cannot";
        }
        String str = super.toString() + " With my waterbending, I " + c + " heal others.";
        return str;
    }

    /**
     * @return boolean
     */
    public boolean isHealer() {
        return healer;
    }

    /**
     * @return string
     */
    public String getName() {
        return name;
    }

    /**
     * @return int
     */
    public int getStrength() {
        return strength;
    }

    /**
     * @return int
     */
    public int getHealth() {
        return health;
    }

    /**
     * @param obj object
     * @return boolean
     */
    public boolean equals(Object obj) {
        if (!(obj instanceof WaterBender)) {
            return false;
        }
        WaterBender other = (WaterBender) obj;
        return (healer == other.healer);
    }

}