/**
*@author Vidit Pokharna
*@version 1.0
*/
public class FireBender extends Bender {
    /**
     *
     */
    private static double fireSourceStrength = 1;

    /**
     * @return double
     */
    public static double getFireSourceStrength() {
        return fireSourceStrength;
    }

    /**
     * @param name string
     * @param strength int
     * @param health int
     */
    public FireBender(String name, int strength, int health) {
        super(name, strength, health);
    }

    /**
     * @param name string
     */
    public FireBender(String name) {
        this(name, 60, 50);
    }

    /**
     * @param b bender
     */
    public void attack(Bender b) {
        if (this.isAlive()) {
            if (b instanceof FireBender) {
                int newHealth = b.getHealth() - ((int) fireSourceStrength);
                if (newHealth < 0) {
                    newHealth = 0;
                }
                b.setHealth(newHealth);
            } else if (b instanceof WaterBender) {
                int newHealth = b.getHealth() - ((int) (strength * fireSourceStrength));
                if (newHealth < 0) {
                    newHealth = 0;
                }
                b.setHealth(newHealth);
            }
        }
        fireSourceStrength -= 0.05;
        if (fireSourceStrength < 0) {
            fireSourceStrength = 0;
        }
    }

    /**
     *
     */
    public void replenishFireSources() {
        if (strength < 50) {
            fireSourceStrength = 0.8;
        } else {
            fireSourceStrength = 1;
        }
    }

    /**
     * @return string
     */
    public String toString() {
        String str = super.toString() + " I bend fire.";
        return str;
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
        if (!(obj instanceof FireBender)) {
            return false;
        }
        return true;
    }
}