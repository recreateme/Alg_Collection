
using System;
using System.Diagnostics;
class Program
{
    static void Main(string[] args)
    {
        string pythonInterpreterPath = @"D:\pt2\python.exe";
        string pythonScriptPath = @"D:\test612\sobel_rec.py";
        if (!File.Exists(pythonInterpreterPath))
        {
            Console.WriteLine("Python interpreter not found at " +  pythonInterpreterPath);
            return;
        }
        if (!File.Exists(pythonScriptPath))
        {
            Console.WriteLine("Python script not found at " + pythonScriptPath);
            return;
        }
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = pythonInterpreterPath,
            Arguments = pythonScriptPath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        try
        {
            using (Process process = new Process())
            {
                process.StartInfo = startInfo;
                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                process.WaitForExit();
                Console.WriteLine(output);
                Console.WriteLine(error);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("An error occurred while running the Python script:");
            Console.WriteLine(ex.Message);
        }
    }
}
