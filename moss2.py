import cv2
import numpy as np

class MOSSE:
    def __init__(self, learning_rate=0.125):
        self.learning_rate = learning_rate
        self.H = None  # فیلتر MOSSE
        self.G = None  # پاسخ ایده‌آل (گاوسی)
        self.window_size = None
        self.eps = 1e-5  # برای جلوگیری از تقسیم بر صفر

    def _create_gaussian_response(self, size):
        """ایجاد پاسخ گاوسی مطلوب برای منطقه هدف"""
        sigma = size[0] / 8
        yy, xx = np.mgrid[-size[0]//2:size[0]//2, -size[1]//2:size[1]//2]
        response = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return response

    def _preprocess(self, img):
        """پیش‌پردازش تصویر"""
        img = np.log(img + 1)  # تبدیل لگاریتمی
        img = (img - img.mean()) / (img.std() + self.eps)  # نرمال‌سازی
        return img * np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]  # پنجره‌گذاری

    def init(self, frame, bbox):
        """مقداردهی اولیه ردیاب با جعبه محدود کننده"""
        x, y, w, h = bbox
        self.window_size = (w, h)
        
        # استخراج منطقه هدف
        target = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        target = target.astype(np.float32)
        
        # ایجاد پاسخ گاوسی
        self.G = np.fft.fft2(self._create_gaussian_response((h, w)))
        
        # پیش‌پردازش و تبدیل فوریه
        F = np.fft.fft2(self._preprocess(target))
        
        # محاسبه فیلتر اولیه
        self.H = (self.G * np.conj(F)) / (F * np.conj(F) + self.eps)
        
        # ایجاد چند نمونه تغییر یافته برای آموزش اولیه
        for _ in range(8):
            # ایجاد نمونه‌های تغییر یافته (جابجایی و چرخش کوچک)
            angle = np.random.uniform(-5, 5)
            scale = np.random.uniform(0.95, 1.05)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
            warped = cv2.warpAffine(target, M, (w, h))
            
            # آموزش با نمونه تغییر یافته
            Fi = np.fft.fft2(self._preprocess(warped))
            Ai = self.G * np.conj(Fi)
            Bi = Fi * np.conj(Fi)
            self.H = (1 - self.learning_rate) * self.H + self.learning_rate * (Ai / (Bi + self.eps))

    def update(self, frame):
        """به‌روزرسانی موقعیت هدف در فریم جدید"""
        if self.H is None:
            raise ValueError("ردیاب مقداردهی اولیه نشده است!")
        
        # استخراج منطقه جستجو (اندازه بزرگتر برای پوشش حرکت)
        h, w = self.window_size
        search_scale = 2
        search_h, search_w = int(h * search_scale), int(w * search_scale)
        
        # تبدیل به خاکستری و پیش‌پردازش
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # محاسبه پاسخ همبستگی
        region = cv2.getRectSubPix(gray, (search_w, search_h), 
                                  (self.pos[0] + w//2, self.pos[1] + h//2))
        region = self._preprocess(region)
        
        # اعمال فیلتر در حوزه فرکانس
        F = np.fft.fft2(region)
        R = np.fft.ifft2(self.H * F)
        r = np.real(R)
        
        # یافتن موقعیت ماکزیمم پاسخ
        _, _, _, max_loc = cv2.minMaxLoc(r)
        
        # محاسبه جابجایی نسبت به مرکز
        dx = max_loc[0] - search_w // 2
        dy = max_loc[1] - search_h // 2
        
        # به‌روزرسانی موقعیت
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        
        # استخراج منطقه هدف جدید برای به‌روزرسانی فیلتر
        x, y = int(self.pos[0]), int(self.pos[1])
        target = gray[y:y+h, x:x+w]
        
        # به‌روزرسانی فیلتر
        if target.shape[0] == h and target.shape[1] == w:
            F_new = np.fft.fft2(self._preprocess(target))
            A_new = self.G * np.conj(F_new)
            B_new = F_new * np.conj(F_new)
            self.H = (1 - self.learning_rate) * self.H + self.learning_rate * (A_new / (B_new + self.eps))
        
        return (x, y, w, h)

# مثال استفاده از ردیاب
if __name__ == "__main__":
    # خواندن ویدیو
    cap = cv2.VideoCapture(0)  # یا مسیر فایل ویدیویی
    
    # خواندن اولین فریم و انتخاب ROI
    ret, frame = cap.read()
    bbox = cv2.selectROI("Select Object", frame, False, False)
    cv2.destroyWindow("Select Object")
    
    # مقداردهی اولیه ردیاب
    tracker = MOSSE(learning_rate=0.1)
    tracker.init(frame, bbox)
    tracker.pos = (bbox[0], bbox[1])  # ذخیره موقعیت اولیه
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # به‌روزرسانی ردیاب
        x, y, w, h = tracker.update(frame)
        
        # رسم جعبه محدود کننده
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("MOSSE Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()