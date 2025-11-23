from django import forms

class DatasetUploadForm(forms.Form):
    # Basic metadata
    name = forms.CharField(max_length=255)
    description = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 3}))

    # Data Source Selection
    data_source = forms.ChoiceField(
        choices=[
            ('file', 'File Upload'),
            ('mongodb', 'MongoDB'),
            ('postgresql', 'PostgreSQL'),
            ('mysql', 'MySQL'),
            ('api', 'API Endpoint'),
        ]
    )

    # -----------------------------
    # FILE UPLOAD
    # -----------------------------
    file = forms.FileField(
        required=False,
        widget=forms.FileInput(attrs={"accept": ".csv,.xlsx,.xls,.json,.txt,.tsv,.parquet"})
    )

    # -----------------------------
    # MongoDB Connection Fields
    # -----------------------------
    mongo_uri = forms.CharField(
        required=False,
        help_text="Leave empty if using host & port fields"
    )
    mongo_host = forms.CharField(required=False)
    mongo_port = forms.IntegerField(required=False)
    mongo_user = forms.CharField(required=False)
    mongo_password = forms.CharField(required=False, widget=forms.PasswordInput(render_value=True))
    mongo_db = forms.CharField(required=False)
    mongo_collection = forms.CharField(required=False)
    mongo_replica_set = forms.CharField(required=False)
    mongo_read_pref = forms.ChoiceField(
        required=False,
        choices=[
            ('primary', 'Primary'),
            ('primaryPreferred', 'Primary Preferred'),
            ('secondary', 'Secondary'),
            ('secondaryPreferred', 'Secondary Preferred'),
            ('nearest', 'Nearest'),
        ]
    )
    mongo_ssl = forms.BooleanField(required=False)

    # -----------------------------
    # PostgreSQL Connection Fields
    # -----------------------------
    pg_host = forms.CharField(required=False)
    pg_port = forms.IntegerField(required=False)
    pg_db = forms.CharField(required=False)
    pg_schema = forms.CharField(required=False)
    pg_user = forms.CharField(required=False)
    pg_password = forms.CharField(required=False, widget=forms.PasswordInput(render_value=True))
    pg_table = forms.CharField(required=False)
    pg_ssl = forms.BooleanField(required=False)
    pg_timeout = forms.IntegerField(required=False)

    # -----------------------------
    # MySQL Connection Fields
    # -----------------------------
    my_host = forms.CharField(required=False)
    my_port = forms.IntegerField(required=False)
    my_db = forms.CharField(required=False)
    my_user = forms.CharField(required=False)
    my_password = forms.CharField(required=False, widget=forms.PasswordInput(render_value=True))
    my_table = forms.CharField(required=False)
    my_charset = forms.CharField(required=False, initial="utf8mb4")
    my_ssl = forms.BooleanField(required=False)
    my_timeout = forms.IntegerField(required=False)

    # -----------------------------
    # API Ingestion
    # -----------------------------
    api_url = forms.URLField(required=False)
    api_method = forms.ChoiceField(
        required=False,
        choices=[
            ('GET', 'GET'),
            ('POST', 'POST'),
            ('PUT', 'PUT'),
            ('PATCH', 'PATCH'),
        ]
    )
    api_headers = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text='JSON formatted headers'
    )
    api_params = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 2}),
        help_text='Query/body params in JSON format'
    )
    api_pagination = forms.CharField(
        required=False,
        help_text="Optional key for paginated API e.g. next_url"
    )
