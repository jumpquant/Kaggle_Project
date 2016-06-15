public class LeapYear {
	public static boolean is_leap(int x) {
		if (x % 400 == 0) {
			return true;
		}
		else if (x%4 ==0 && x%100 != 0) {
			return true;
		}
		return false;
	}
	public static void main(String[] args) {
		int year = 2000;
		if (is_leap(year)) {
			System.out.println(year + " is a leap year");
		}
		else {
			System.out.println(year + " is not a leap year");
		}
	}
}