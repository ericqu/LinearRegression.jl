@testset "Cook's Distance" begin 
        st_df = DataFrame( 
                Y=[6.4, 7.4, 10.4, 15.1, 12.3 , 11.4],
                XA=[1.5, 6.5, 11.5, 19.9, 17.0, 15.5],
                XB=[1.8, 7.8, 11.8, 20.5, 17.3, 15.8], 
                XC=[3., 13., 23., 39.8, 34., 31.],
                # values from SAS proc reg
                CooksD_base=[1.4068501943, 0.176809102, 0.0026655177, 1.0704009915, 0.0875726457, 0.1331183932], 
                CooksD_multi=[1.7122291956, 18.983407026, 0.000118078, 0.8470797843, 0.0715921999, 0.1105843157],
                )
       
        t_lm_base = regress(@formula(Y ~ 1+ XA), st_df)
        results = predict_and_stats(t_lm_base, st_df, req_stats=["all"])
        @test isapprox(st_df.CooksD_base, results.cooksd)

        t_lm_multi = regress(@formula(Y ~ 1+ XA + XB), st_df)
        results = predict_and_stats(t_lm_multi, st_df, req_stats=["all"])
        @test isapprox(st_df.CooksD_multi, results.cooksd)
end