#ifndef COLOR_H
#define COLOR_H

#ifdef _WIN32

#define RESET       "\\E[0;0m"
#define BLACK       "\\E[0;30m"             /* Black */
#define RED         "\\E[0;31m"             /* Red */
#define GREEN       "\\E[0;32m"             /* Green */
#define YELLOW      "\\E[0;33m"             /* Yellow */
#define BLUE        "\\E[0;34m"             /* Blue */
#define MAGENTA     "\\E[0;35m"             /* Magenta */
#define CYAN        "\\E[0;36m"             /* Cyan */
#define WHITE       "\\E[0;37m"             /* White */
#define REDPURPLE   "\\E[0;95m"             /* Red Purple */
#define BOLDBLACK   "\\E[0;1m\\E[0;30m"      /* Bold Black */
#define BOLDRED     "\\E[0;1m\\E[0;31m"      /* Bold Red */
#define BOLDGREEN   "\\E[0;1m\\E[0;32m"      /* Bold Green */
#define BOLDYELLOW  "\\E[0;1m\\E[0;33m"      /* Bold Yellow */
#define BOLDBLUE    "\\E[0;1m\\E[0;34m"      /* Bold Blue */
#define BOLDMAGENTA "\\E[0;1m\\E[0;35m"      /* Bold Magenta */
#define BOLDCYAN    "\\E[0;1m\\E[0;36m"      /* Bold Cyan */
#define BOLDWHITE   "\\E[0;1m\\E[0;37m"      /* Bold White */
#define BOLDREDPURPLE   "\\E[0;1m\\E[0;95m"  /* Bold Red Purple */

#else

#define RESET       "\033[0m"
#define BLACK       "\033[30m"             /* Black */
#define RED         "\033[31m"             /* Red */
#define GREEN       "\033[32m"             /* Green */
#define YELLOW      "\033[33m"             /* Yellow */
#define BLUE        "\033[34m"             /* Blue */
#define MAGENTA     "\033[35m"             /* Magenta */
#define CYAN        "\033[36m"             /* Cyan */
#define WHITE       "\033[37m"             /* White */
#define REDPURPLE   "\033[95m"             /* Red Purple */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#define BOLDREDPURPLE   "\033[1m\033[95m"  /* Bold Red Purple */

#endif // _WIN32

/// Represent OpenGL floating point colour: Red, Green and Blue with alpha.
struct Colour
{
    inline static Colour White()
    {
        return Colour(1.0f, 1.0f, 1.0f, 1.0f);
    }

    inline static Colour Black()
    {
        return Colour(0.0f, 0.0f, 0.0f, 1.0f);
    }

    inline static Colour Red()
    {
        return Colour(1.0f, 0.0f, 0.0f, 1.0f);
    }

    inline static Colour Green()
    {
        return Colour(0.0f, 1.0f, 0.0f, 1.0f);
    }

    inline static Colour Blue()
    {
        return Colour(0.0f, 0.0f, 1.0f, 1.0f);
    }

    inline static Colour Unspecified()
    {
        return Colour(
                std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()
        );
    }

    /// Default constructs white.
    inline Colour()
            : red(1.0f), green(1.0f), blue(1.0f), alpha(1.0f)
    {
    }

    /// Construct from component values
    inline Colour(const float red, const float green, const float blue, const float alpha = 1.0f)
            : red(red), green(green), blue(blue), alpha(alpha)
    {
    }

    /// Construct from rgba array.
    inline Colour(const float rgba[4])
    {
        r = rgba[0];
        g = rgba[1];
        b = rgba[2];
        a = rgba[3];
    }

    /// Return pointer to OpenGL compatible RGBA array.
    inline float *Get()
    {
        return c;
    }

    /// Return this colour with alpha adjusted.
    inline Colour WithAlpha(const float alpha)
    {
        return Colour(r, g, b, alpha);
    }

    /// Construct from HSV Colour
    /// @param hue Colour hue in range [0,1]
    /// @param sat Saturation in range [0,1]
    /// @param val Value / Brightness in range [0,1].
    static inline Colour Hsv(const float hue, const float sat = 1.0f, const float val = 1.0f, const float alpha = 1.0f)
    {
        const float h = 6.0f * hue;
        const int i = (int) floor(h);
        const float f = (i % 2 == 0) ? 1 - (h - i) : h - i;
        const float m = val * (1 - sat);
        const float n = val * (1 - sat * f);

        switch (i)
        {
            case 0:
                return Colour(val, n, m, alpha);
            case 1:
                return Colour(n, val, m, alpha);
            case 2:
                return Colour(m, val, n, alpha);
            case 3:
                return Colour(m, n, val, alpha);
            case 4:
                return Colour(n, m, val, alpha);
            case 5:
                return Colour(val, m, n, alpha);
            default:
                throw std::runtime_error("Found extra colour in rainbow.");
        }
    }

    union
    {
        struct
        {
            float red;
            float green;
            float blue;
            float alpha;
        };
        struct
        {
            float r;
            float g;
            float b;
            float a;
        };
        float c[4];
    };

};

class ColourWheel
{
public:
    /// Construct ColourWheel with Saturation, Value and Alpha constant.
    inline ColourWheel(float saturation = 0.5f, float value = 1.0f, float alpha = 1.0f)
            : unique_colours(0), sat(saturation), val(value), alpha(alpha)
    {

    }

    /// Use Golden ratio (/angle) to pick well spaced colours.
    inline Colour GetColourBin(int i) const
    {
        float hue = i * 0.5f * (3.0f - sqrt(5.0f));
        hue -= (int) hue;
        return Colour::Hsv(hue, sat, val, alpha);
    }

    /// Return next unique colour from ColourWheel.
    inline Colour GetUniqueColour()
    {
        return GetColourBin(unique_colours++);
    }

    /// Reset colour wheel counter to initial state
    inline void Reset()
    {
        unique_colours = 0;
    }

protected:
    int unique_colours;
    float sat;
    float val;
    float alpha;
};

#endif
